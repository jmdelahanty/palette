from __future__ import annotations

import copy
import json

import pyarrow as pa
import pytest

from fisheye.baseline_strategy.contracts import (
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_ARROW_CONTRACTS,
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
    BASELINE_STRATEGY_FEATURES_TABLE,
    BASELINE_STRATEGY_PRIMARY_KEYS,
    BASELINE_STRATEGY_TABLES,
    LEGACY_SCHEMA_VERSION,
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    baseline_strategy_arrow_contract_envelope,
    exact_baseline_strategy_arrow_schema,
    normalize_baseline_strategy_row,
    normalize_baseline_strategy_rows,
    validate_baseline_strategy_arrow_contract_envelope,
    validate_baseline_strategy_arrow_schema,
    validate_baseline_strategy_primary_keys,
)


EXPECTED_FIELD_NAMES = {
    BASELINE_STRATEGY_FEATURES_TABLE: (
        "schema_id",
        "schema_version",
        "table_name",
        "method",
        "method_version",
        "recording_id",
        "session_id",
        "subject_id",
        "track_id",
        "baseline_window_id",
        "baseline_window_label",
        "source_export_run_id",
        "zarr_path",
        "sample_features_available",
        "time_bin_features_available",
        "duration_s",
        "tracking_dropout_fraction",
        "mean_speed_mm_s",
        "median_speed_mm_s",
        "p95_speed_mm_s",
        "total_path_mm",
        "path_per_min_mm",
        "bout_count",
        "bout_rate_per_min",
        "wall_fraction",
        "wall_band_mm",
        "experimental_area_geometry_type",
        "boundary_distance_method",
        "wall_fraction_denominator",
        "active_wall_fraction_denominator",
        "expected_uniform_wall_fraction",
        "wall_enrichment_ratio",
        "wall_enrichment_log2",
        "mean_center_distance_norm",
        "source_spatial_entropy_normalized",
        "source_quadrant_entropy_normalized",
        "source_spatial_max_cell_fraction",
        "source_quadrant_max_fraction",
        "active_speed_threshold_mm_s",
        "spatial_grid_size",
        "dwell_grid_size",
        "time_bin_count",
        "wall_fraction_early",
        "wall_fraction_late",
        "wall_fraction_delta_late_minus_early",
        "wall_fraction_slope_per_baseline",
        "mean_speed_mm_s_early",
        "mean_speed_mm_s_late",
        "mean_speed_mm_s_delta_late_minus_early",
        "mean_speed_mm_s_slope_per_baseline",
        "distance_travelled_mm_early",
        "distance_travelled_mm_late",
        "distance_travelled_mm_delta_late_minus_early",
        "distance_travelled_mm_slope_per_baseline",
        "center_distance_norm_early",
        "center_distance_norm_late",
        "center_distance_norm_delta_late_minus_early",
        "center_distance_norm_slope_per_baseline",
        "bout_count_early",
        "bout_count_late",
        "bout_count_delta_late_minus_early",
        "bout_count_slope_per_baseline",
        "valid_sample_count",
        "active_sample_fraction",
        "boundary_distance_sample_source",
        "active_wall_fraction",
        "occupancy_accessible_cell_count",
        "occupancy_visited_cell_count",
        "occupancy_visited_cell_fraction",
        "occupancy_coverage_fraction",
        "occupancy_entropy_accessible_normalized",
        "occupancy_js_divergence_uniform",
        "occupancy_uniform_reference",
        "occupancy_max_cell_fraction",
        "latency_to_half_final_coverage_s",
        "dominant_dwell_cell_fraction",
        "dominant_to_second_dwell_ratio",
        "dominant_dwell_visit_count",
        "dominant_dwell_center_distance_norm",
        "exploration_episode_count",
        "wall_following_episode_fraction",
        "dominant_dwell_excursion_count",
        "dominant_dwell_return_fraction",
        "median_episode_path_length_mm",
        "median_episode_tortuosity",
        "feature_status",
        "feature_reason",
        "analysis_run_id",
    ),
    BASELINE_EXPLORATION_EPISODES_TABLE: (
        "schema_id",
        "schema_version",
        "table_name",
        "method",
        "method_version",
        "recording_id",
        "session_id",
        "subject_id",
        "track_id",
        "baseline_window_id",
        "baseline_window_label",
        "source_export_run_id",
        "zarr_path",
        "episode_id",
        "episode_start_s",
        "episode_end_s",
        "episode_duration_s",
        "sample_count",
        "path_length_mm",
        "net_displacement_mm",
        "tortuosity",
        "minimum_center_distance_mm",
        "maximum_inward_excursion_mm",
        "wall_sample_fraction",
        "mean_tangential_alignment",
        "wall_following",
        "origin_dominant_dwell_zone",
        "destination_dominant_dwell_zone",
        "returned_to_dominant_dwell_zone",
        "path_length_method",
        "boundary_distance_method",
        "analysis_run_id",
    ),
    BASELINE_STRATEGY_CLASSIFICATION_TABLE: (
        "schema_id",
        "schema_version",
        "table_name",
        "method",
        "method_version",
        "recording_id",
        "session_id",
        "subject_id",
        "track_id",
        "baseline_window_id",
        "baseline_window_label",
        "source_export_run_id",
        "zarr_path",
        "activity_score",
        "activity_metric_count",
        "boundary_score",
        "boundary_metric_count",
        "spatial_distribution_score",
        "spatial_distribution_metric_count",
        "home_base_score",
        "home_base_metric_count",
        "temporal_expansion_score",
        "temporal_expansion_metric_count",
        "classification_status",
        "classification_reason",
        "reference_scope",
        "relative_score_threshold",
        "activity_state",
        "boundary_strategy",
        "home_base_state",
        "spatial_organization",
        "temporal_pattern",
        "primary_strategy",
        "classification_confidence_score",
        "confidence_semantics",
        "anxiety_inference_permitted",
        "analysis_run_id",
    ),
    BASELINE_STRATEGY_CLUSTERS_TABLE: (
        "schema_id",
        "schema_version",
        "table_name",
        "method",
        "method_version",
        "recording_id",
        "session_id",
        "subject_id",
        "track_id",
        "baseline_window_id",
        "baseline_window_label",
        "source_export_run_id",
        "zarr_path",
        "cluster_status",
        "cluster_reason",
        "cluster_id",
        "cluster_probability",
        "cluster_probability_threshold",
        "selected_component_count",
        "selected_bic",
        "bic_by_component_count_json",
        "cluster_stability_median_ari",
        "cluster_stability_resample_count",
        "cluster_axes",
        "cluster_semantics",
        "analysis_run_id",
    ),
}

# These digests pin order, Arrow type, nullability, namespace, and table schema
# version independently from the human-readable field-name snapshot above.
EXPECTED_CONTRACT_DIGESTS = {
    BASELINE_STRATEGY_FEATURES_TABLE: (
        "8d3ab3b68ed34ecc16e8c316b3a5f891439e3df3a0d62e8bd6831267eb831d4b"
    ),
    BASELINE_EXPLORATION_EPISODES_TABLE: (
        "559e1fa497b8feef0df47835f4e81578f1cbd3a83a0afd9c485f7b5b53fe916b"
    ),
    BASELINE_STRATEGY_CLASSIFICATION_TABLE: (
        "e34771ba686645180faca9d15370b8fac4d564eafb9b92ec9ed34959a5ccffdb"
    ),
    BASELINE_STRATEGY_CLUSTERS_TABLE: (
        "63671d58146536662503d0e14441fa8a00e155983843af5745b55bf4e84c6edc"
    ),
}


def _minimal_row(table_name: str, *, recording_id: str = "recording_001") -> dict:
    contract = BASELINE_STRATEGY_ARROW_CONTRACTS[table_name]
    row = {}
    for field in contract.fields:
        if field.nullable:
            continue
        if field.arrow_type == "string":
            row[field.name] = f"{field.name}_value"
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.0
        elif field.arrow_type == "bool":
            row[field.name] = True
        elif field.arrow_type == "list<string>":
            row[field.name] = ["value"]
        else:  # pragma: no cover - the installed contract snapshot is exhaustive.
            raise AssertionError(field.arrow_type)
    row.update(
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        table_name=table_name,
        method=METHOD,
        method_version=METHOD_VERSION,
        recording_id=recording_id,
        track_id=0,
        baseline_window_id=0,
        analysis_run_id="strategy_001",
    )
    if table_name == BASELINE_EXPLORATION_EPISODES_TABLE:
        row["episode_id"] = 0
    return row


def test_v2_contract_inventory_and_exact_field_snapshots() -> None:
    assert LEGACY_SCHEMA_VERSION == 1
    assert SCHEMA_VERSION == 2
    assert tuple(BASELINE_STRATEGY_ARROW_CONTRACTS) == BASELINE_STRATEGY_TABLES
    assert set(BASELINE_STRATEGY_PRIMARY_KEYS) == set(BASELINE_STRATEGY_TABLES)
    for table_name in BASELINE_STRATEGY_TABLES:
        contract = BASELINE_STRATEGY_ARROW_CONTRACTS[table_name]
        assert tuple(field.name for field in contract.fields) == EXPECTED_FIELD_NAMES[
            table_name
        ]
        assert contract.schema_version == SCHEMA_VERSION
        assert contract.primary_key == BASELINE_STRATEGY_PRIMARY_KEYS[table_name]
        assert contract.payload_sha256 == EXPECTED_CONTRACT_DIGESTS[table_name]


def test_v2_contract_envelope_is_closed_all_exact_and_rebuilt() -> None:
    envelope = baseline_strategy_arrow_contract_envelope()
    assert tuple(envelope["exact_tables"]) == BASELINE_STRATEGY_TABLES
    assert envelope["inferred_v2_compatibility_tables"] == []
    assert validate_baseline_strategy_arrow_contract_envelope(envelope) == envelope

    tampered = copy.deepcopy(envelope)
    tampered["exact_tables"][BASELINE_STRATEGY_FEATURES_TABLE]["fields"][0][
        "nullable"
    ] = True
    with pytest.raises(ValueError, match="payload digest|installed contracts"):
        validate_baseline_strategy_arrow_contract_envelope(tampered)


@pytest.mark.parametrize("table_name", BASELINE_STRATEGY_TABLES)
def test_exact_schema_supports_typed_rows_and_exact_empty_tables(table_name: str) -> None:
    row = normalize_baseline_strategy_row(table_name, _minimal_row(table_name))
    schema = exact_baseline_strategy_arrow_schema(table_name)
    validate_baseline_strategy_arrow_schema(table_name, schema)
    assert pa.Table.from_pylist([row], schema=schema).schema == schema
    assert pa.Table.from_pylist([], schema=schema).schema == schema


@pytest.mark.parametrize("table_name", BASELINE_STRATEGY_TABLES)
def test_row_normalization_fills_nullable_fields_in_exact_order(table_name: str) -> None:
    normalized = normalize_baseline_strategy_row(table_name, _minimal_row(table_name))
    assert tuple(normalized) == EXPECTED_FIELD_NAMES[table_name]
    for field in BASELINE_STRATEGY_ARROW_CONTRACTS[table_name].fields:
        if field.nullable:
            assert normalized[field.name] is None


def test_row_normalization_rejects_open_or_incomplete_rows() -> None:
    row = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE)
    row["future_metric"] = 1.0
    with pytest.raises(ValueError, match="unexpected fields"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_FEATURES_TABLE, row)

    row = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE)
    del row["feature_status"]
    with pytest.raises(ValueError, match="required field is missing"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_FEATURES_TABLE, row)

    row = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE)
    row["feature_status"] = None
    with pytest.raises(ValueError, match="required field is null"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_FEATURES_TABLE, row)


def test_row_normalization_binds_publication_run_identity() -> None:
    row = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE)
    del row["analysis_run_id"]
    normalized = normalize_baseline_strategy_row(
        BASELINE_STRATEGY_FEATURES_TABLE,
        row,
        analysis_run_id="strategy_002",
    )
    assert normalized["analysis_run_id"] == "strategy_002"

    row["analysis_run_id"] = "wrong"
    with pytest.raises(ValueError, match="differs from the publication run"):
        normalize_baseline_strategy_row(
            BASELINE_STRATEGY_FEATURES_TABLE,
            row,
            analysis_run_id="strategy_002",
        )


def test_row_normalization_is_the_explicit_v1_to_v2_boundary() -> None:
    row = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE)
    row["schema_version"] = LEGACY_SCHEMA_VERSION
    normalized = normalize_baseline_strategy_row(BASELINE_STRATEGY_FEATURES_TABLE, row)
    assert normalized["schema_version"] == SCHEMA_VERSION

    row["schema_version"] = 3
    with pytest.raises(ValueError, match="schema_version must equal"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_FEATURES_TABLE, row)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    (
        ("schema_id", "wrong", "schema_id must equal"),
        ("table_name", "wrong", "table_name must equal"),
        ("method", "wrong", "method must equal"),
        ("method_version", "wrong", "method_version must equal"),
        ("recording_id", 1, "recording_id must be a string"),
        ("sample_features_available", 1, "must be a bool"),
        ("spatial_grid_size", True, "must be an integer"),
        ("spatial_grid_size", 2**63, "outside the int64 range"),
        ("duration_s", float("nan"), "must be a finite number"),
        ("duration_s", float("inf"), "must be a finite number"),
    ),
)
def test_row_normalization_rejects_wrong_constants_types_and_ranges(
    field_name: str,
    bad_value: object,
    message: str,
) -> None:
    row = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE)
    row[field_name] = bad_value
    with pytest.raises(ValueError, match=message):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_FEATURES_TABLE, row)


def test_cluster_axes_require_a_list_of_strings() -> None:
    row = _minimal_row(BASELINE_STRATEGY_CLUSTERS_TABLE)
    row["cluster_axes"] = ["activity_score", 1]
    with pytest.raises(ValueError, match="must be a list of strings"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_CLUSTERS_TABLE, row)


def test_cluster_legacy_bic_mapping_becomes_canonical_json() -> None:
    row = _minimal_row(BASELINE_STRATEGY_CLUSTERS_TABLE)
    row["bic_by_component_count"] = {"2": 12, "1": 9.5}
    normalized = normalize_baseline_strategy_row(BASELINE_STRATEGY_CLUSTERS_TABLE, row)
    assert "bic_by_component_count" not in normalized
    assert normalized["bic_by_component_count_json"] == '{"1":9.5,"2":12.0}'
    assert json.loads(normalized["bic_by_component_count_json"]) == {
        "1": 9.5,
        "2": 12.0,
    }

    noncanonical = _minimal_row(BASELINE_STRATEGY_CLUSTERS_TABLE)
    noncanonical["bic_by_component_count_json"] = '{"2": 12, "1": 9.5}'
    with pytest.raises(ValueError, match="not canonical JSON"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_CLUSTERS_TABLE, noncanonical)

    duplicate = _minimal_row(BASELINE_STRATEGY_CLUSTERS_TABLE)
    duplicate["bic_by_component_count_json"] = '{"1":1.0,"1":2.0}'
    with pytest.raises(ValueError, match="duplicate component keys"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_CLUSTERS_TABLE, duplicate)

    both = _minimal_row(BASELINE_STRATEGY_CLUSTERS_TABLE)
    both["bic_by_component_count"] = {"1": 1.0}
    both["bic_by_component_count_json"] = '{"1":1.0}'
    with pytest.raises(ValueError, match="cannot coexist"):
        normalize_baseline_strategy_row(BASELINE_STRATEGY_CLUSTERS_TABLE, both)


def test_primary_keys_reject_null_and_duplicate_identities() -> None:
    first = normalize_baseline_strategy_row(
        BASELINE_STRATEGY_FEATURES_TABLE,
        _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE),
    )
    duplicate = dict(first)
    with pytest.raises(ValueError, match="duplicate primary key"):
        validate_baseline_strategy_primary_keys(
            BASELINE_STRATEGY_FEATURES_TABLE,
            [first, duplicate],
        )

    null_key = dict(first)
    null_key["recording_id"] = None
    with pytest.raises(ValueError, match="null or empty primary-key"):
        validate_baseline_strategy_primary_keys(
            BASELINE_STRATEGY_FEATURES_TABLE,
            [null_key],
        )

    second = _minimal_row(BASELINE_STRATEGY_FEATURES_TABLE, recording_id="recording_002")
    normalized = normalize_baseline_strategy_rows(
        BASELINE_STRATEGY_FEATURES_TABLE,
        [first, second],
    )
    assert len(normalized) == 2


def test_episode_primary_key_includes_episode_id() -> None:
    first = normalize_baseline_strategy_row(
        BASELINE_EXPLORATION_EPISODES_TABLE,
        _minimal_row(BASELINE_EXPLORATION_EPISODES_TABLE),
    )
    second = dict(first)
    second["episode_id"] = 1
    validate_baseline_strategy_primary_keys(
        BASELINE_EXPLORATION_EPISODES_TABLE,
        [first, second],
    )
