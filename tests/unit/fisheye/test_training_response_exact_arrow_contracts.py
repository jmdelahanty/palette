from __future__ import annotations

from copy import deepcopy

import pyarrow as pa
import pytest

from fisheye.training_response.contracts import (
    ARROW_TABLE_CONTRACTS,
    LEGACY_SCHEMA_VERSION,
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
    TRAINING_RESPONSE_FEATURES_TABLE,
    TRAINING_RESPONSE_TABLES,
    TrainingResponseConfig,
    exact_training_response_arrow_schema,
    normalize_training_response_row,
    normalize_training_response_rows,
    training_response_arrow_contract_envelope,
    validate_training_response_arrow_contract_envelope,
    validate_training_response_primary_keys,
)


COMMON_SIGNATURE = (
    ("schema_id", "string", False),
    ("schema_version", "int32", False),
    ("table_name", "string", False),
    ("method", "string", False),
    ("method_version", "string", False),
    ("recording_id", "string", False),
    ("training_window_id", "int64", True),
    ("source_export_run_id", "string", False),
    ("protocol_name", "string", True),
)

FEATURE_FLOAT_NAMES = (
    "pre_duration_s",
    "training_duration_s",
    "pre_tracking_dropout_fraction",
    "training_tracking_dropout_fraction",
    "pre_valid_position_fraction",
    "training_valid_position_fraction",
    "pre_path_per_min_mm",
    "training_path_per_min_mm",
    "path_per_min_log2_ratio",
    "pre_mean_speed_mm_s",
    "training_mean_speed_mm_s",
    "mean_speed_mm_s_log2_ratio",
    "pre_p95_speed_mm_s",
    "training_p95_speed_mm_s",
    "p95_speed_mm_s_log2_ratio",
    "pre_bout_rate_per_min",
    "training_bout_rate_per_min",
    "bout_rate_per_min_log2_ratio",
    "pre_mean_bout_path_length_mm",
    "training_mean_bout_path_length_mm",
    "mean_bout_path_length_mm_log2_ratio",
    "pre_mean_abs_bout_net_heading_change_deg",
    "training_mean_abs_bout_net_heading_change_deg",
    "mean_abs_bout_net_heading_change_deg_log2_ratio",
    "arena_radius_mm",
    "pre_wall_fraction",
    "training_wall_fraction",
    "wall_fraction_delta",
    "pre_center_distance_norm",
    "training_center_distance_norm",
    "center_distance_norm_delta",
    "aggressive_pre_p05_distance_mm",
    "aggressive_training_p05_distance_mm",
    "aggressive_p05_distance_mm_delta",
    "aggressive_pre_p50_distance_mm",
    "aggressive_training_p50_distance_mm",
    "aggressive_p50_distance_mm_delta",
    "aggressive_pre_fraction_within_threshold",
    "aggressive_training_fraction_within_threshold",
    "aggressive_fraction_within_threshold_delta",
    "aggressive_threshold_mm",
    "aggressive_pre_mean_alignment_cos",
    "aggressive_training_mean_alignment_cos",
    "aggressive_mean_alignment_cos_delta",
    "aggressive_pre_fraction_front_45",
    "aggressive_training_fraction_front_45",
    "aggressive_fraction_front_45_delta",
    "aggressive_pre_fraction_behind_45",
    "aggressive_training_fraction_behind_45",
    "aggressive_fraction_behind_45_delta",
    "aggressive_pre_circular_resultant_length",
    "aggressive_training_circular_resultant_length",
    "aggressive_circular_resultant_length_delta",
    "aggressive_training_near_speed_mm_s",
    "aggressive_training_far_speed_mm_s",
    "aggressive_near_minus_far_speed_mm_s",
    "inert_pre_p05_distance_mm",
    "inert_training_p05_distance_mm",
    "inert_p05_distance_mm_delta",
    "inert_pre_p50_distance_mm",
    "inert_training_p50_distance_mm",
    "inert_p50_distance_mm_delta",
    "inert_pre_fraction_within_threshold",
    "inert_training_fraction_within_threshold",
    "inert_fraction_within_threshold_delta",
    "inert_threshold_mm",
    "inert_pre_mean_alignment_cos",
    "inert_training_mean_alignment_cos",
    "inert_mean_alignment_cos_delta",
    "inert_pre_fraction_front_45",
    "inert_training_fraction_front_45",
    "inert_fraction_front_45_delta",
    "inert_pre_fraction_behind_45",
    "inert_training_fraction_behind_45",
    "inert_fraction_behind_45_delta",
    "inert_pre_circular_resultant_length",
    "inert_training_circular_resultant_length",
    "inert_circular_resultant_length_delta",
    "inert_training_near_speed_mm_s",
    "inert_training_far_speed_mm_s",
    "inert_near_minus_far_speed_mm_s",
    "training_role_p50_distance_contrast_mm",
    "training_role_p05_distance_contrast_mm",
    "training_role_within_threshold_contrast",
)

FEATURE_SIGNATURE = (
    COMMON_SIGNATURE
    + (
        ("pre_window_label", "string", False),
        ("training_window_label", "string", False),
        ("zarr_path", "string", True),
        ("temporal_training_features_available", "bool", False),
        ("temporal_training_feature_reason", "string", False),
    )
    + tuple((name, "float64", True) for name in FEATURE_FLOAT_NAMES)
    + (
        ("feature_status", "string", False),
        ("feature_reason", "string", True),
        ("interpretation_guardrail", "string", False),
        ("analysis_run_id", "string", False),
    )
)

CLASSIFICATION_SIGNATURE = COMMON_SIGNATURE + (
    ("zarr_path", "string", True),
    ("locomotor_response_score", "float64", True),
    ("locomotor_response_metric_count", "int64", False),
    ("boundary_response_score", "float64", True),
    ("boundary_response_metric_count", "int64", False),
    ("aggressive_proximity_score", "float64", True),
    ("aggressive_proximity_metric_count", "int64", False),
    ("role_distance_selectivity_score", "float64", True),
    ("role_distance_selectivity_metric_count", "int64", False),
    ("close_contact_vigor_score", "float64", True),
    ("close_contact_vigor_metric_count", "int64", False),
    ("classification_status", "string", False),
    ("classification_reason", "string", True),
    ("reference_scope", "string", False),
    ("relative_score_threshold", "float64", False),
    ("locomotor_response", "string", False),
    ("boundary_response", "string", False),
    ("aggressive_proximity_state", "string", False),
    ("role_distance_selectivity", "string", False),
    ("close_contact_vigor", "string", False),
    ("primary_training_profile", "string", False),
    ("profile_separation_semantics", "string", False),
    ("causal_avoidance_inference_permitted", "bool", False),
    ("temporal_adaptation_inference_permitted", "bool", False),
    ("profile_separation_score", "float64", True),
    ("analysis_run_id", "string", False),
)

CLUSTER_SIGNATURE = COMMON_SIGNATURE + (
    ("zarr_path", "string", True),
    ("cluster_status", "string", False),
    ("cluster_reason", "string", True),
    ("cluster_id", "int64", True),
    ("cluster_probability", "float64", True),
    ("cluster_probability_threshold", "float64", False),
    ("selected_component_count", "int64", True),
    ("selected_bic", "float64", True),
    ("bic_by_component_count_json", "string", True),
    ("cluster_stability_median_ari", "float64", True),
    ("cluster_stability_threshold", "float64", False),
    ("cluster_stability_resample_count", "int64", False),
    ("cluster_min_rows_per_component", "int64", False),
    ("cluster_axes", "list<string>", False),
    ("cluster_semantics", "string", False),
    ("analysis_run_id", "string", False),
)

EXPECTED_CONTRACT_DIGESTS = {
    TRAINING_RESPONSE_FEATURES_TABLE: (
        "d7ede0b4d814466f977d1cd4a9e102ecbd6bae07f809b0dbe5ae23bfba7a4c65"
    ),
    TRAINING_RESPONSE_CLASSIFICATION_TABLE: (
        "c6c41c1e8b2479c6c1545a22065f352a82914179d0b7a0ff20952eb2021ef755"
    ),
    TRAINING_RESPONSE_CLUSTERS_TABLE: (
        "6e307734cadc30036cfdc735433323584fb663174b72ec8d346bf24f8de5fedc"
    ),
}


def _signature(table_name: str) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (item.name, item.arrow_type, item.nullable)
        for item in ARROW_TABLE_CONTRACTS[table_name].fields
    )


def _minimal_row(table_name: str, *, recording_id: str = "recording_1") -> dict:
    row: dict[str, object] = {}
    for item in ARROW_TABLE_CONTRACTS[table_name].fields:
        if item.nullable or item.name == "analysis_run_id":
            continue
        if item.arrow_type == "string":
            row[item.name] = "value"
        elif item.arrow_type in {"int32", "int64"}:
            row[item.name] = 1
        elif item.arrow_type == "float64":
            row[item.name] = 1.0
        elif item.arrow_type == "bool":
            row[item.name] = False
        elif item.arrow_type == "list<string>":
            row[item.name] = ["value"]
        else:  # pragma: no cover - the snapshot above closes this vocabulary.
            raise AssertionError(item.arrow_type)
    row.update(
        schema_id=SCHEMA_ID,
        schema_version=LEGACY_SCHEMA_VERSION,
        table_name=table_name,
        method=METHOD,
        method_version=METHOD_VERSION,
        recording_id=recording_id,
        source_export_run_id="source_1",
    )
    return row


def test_training_response_v2_freezes_all_three_ordered_schemas() -> None:
    assert SCHEMA_VERSION == 2
    assert LEGACY_SCHEMA_VERSION == 1
    assert _signature(TRAINING_RESPONSE_FEATURES_TABLE) == FEATURE_SIGNATURE
    assert _signature(TRAINING_RESPONSE_CLASSIFICATION_TABLE) == (
        CLASSIFICATION_SIGNATURE
    )
    assert _signature(TRAINING_RESPONSE_CLUSTERS_TABLE) == CLUSTER_SIGNATURE
    assert {name: len(contract.fields) for name, contract in ARROW_TABLE_CONTRACTS.items()} == {
        TRAINING_RESPONSE_FEATURES_TABLE: 102,
        TRAINING_RESPONSE_CLASSIFICATION_TABLE: 35,
        TRAINING_RESPONSE_CLUSTERS_TABLE: 25,
    }
    for table_name, contract in ARROW_TABLE_CONTRACTS.items():
        assert contract.schema_version == SCHEMA_VERSION
        assert contract.primary_key == ("analysis_run_id", "recording_id")
        assert contract.payload_sha256 == EXPECTED_CONTRACT_DIGESTS[table_name]


def test_training_response_envelope_is_closed_exact_and_digest_bound() -> None:
    envelope = training_response_arrow_contract_envelope()
    assert set(envelope["exact_tables"]) == set(TRAINING_RESPONSE_TABLES)
    assert envelope["inferred_v2_compatibility_tables"] == []
    assert {
        value["schema_version"] for value in envelope["exact_tables"].values()
    } == {SCHEMA_VERSION}
    assert {
        tuple(value["primary_key"])
        for value in envelope["exact_tables"].values()
    } == {("analysis_run_id", "recording_id")}
    assert validate_training_response_arrow_contract_envelope(envelope) == envelope

    tampered = deepcopy(envelope)
    first = tampered["exact_tables"][TRAINING_RESPONSE_FEATURES_TABLE]["fields"][0]
    first["nullable"] = True
    with pytest.raises(ValueError, match="digest|installed contracts"):
        validate_training_response_arrow_contract_envelope(tampered)


@pytest.mark.parametrize("table_name", TRAINING_RESPONSE_TABLES)
def test_exact_schema_supports_typed_zero_row_tables(table_name: str) -> None:
    schema = exact_training_response_arrow_schema(table_name, metadata={})
    empty = pa.Table.from_pylist([], schema=schema)
    assert empty.num_rows == 0
    assert empty.schema.remove_metadata() == schema.remove_metadata()
    assert empty.schema.metadata[b"palette.arrow_schema_mode"] == b"exact"
    assert empty.schema.metadata[b"palette.arrow_schema_version"] == b"2"


@pytest.mark.parametrize("table_name", TRAINING_RESPONSE_TABLES)
def test_normalization_completes_status_independent_v2_rows(table_name: str) -> None:
    row = _minimal_row(table_name)
    for name in (
        "interpretation_guardrail",
        "reference_scope",
        "relative_score_threshold",
        "cluster_probability_threshold",
        "cluster_stability_threshold",
        "cluster_stability_resample_count",
        "cluster_min_rows_per_component",
        "cluster_axes",
        "cluster_semantics",
    ):
        row.pop(name, None)
    normalized = normalize_training_response_rows(
        table_name,
        [row],
        analysis_run_id="analysis_1",
    )[0]
    assert tuple(normalized) == tuple(
        item.name for item in ARROW_TABLE_CONTRACTS[table_name].fields
    )
    assert normalized["schema_version"] == SCHEMA_VERSION
    assert normalized["analysis_run_id"] == "analysis_1"
    assert normalized["training_window_id"] is None
    pa.Table.from_pylist(
        [normalized],
        schema=exact_training_response_arrow_schema(table_name, metadata={}),
    )


def test_cluster_normalizer_replaces_dynamic_bic_struct_with_canonical_json() -> None:
    row = _minimal_row(TRAINING_RESPONSE_CLUSTERS_TABLE)
    row["bic_by_component_count"] = {"2": 5, "1": 10.0}
    normalized = normalize_training_response_row(
        TRAINING_RESPONSE_CLUSTERS_TABLE,
        row,
        analysis_run_id="analysis_1",
    )
    assert "bic_by_component_count" not in normalized
    assert normalized["bic_by_component_count_json"] == '{"1":10.0,"2":5.0}'

    row["bic_by_component_count_json"] = '{"1":1.0}'
    with pytest.raises(ValueError, match="both legacy and v2"):
        normalize_training_response_row(
            TRAINING_RESPONSE_CLUSTERS_TABLE,
            row,
            analysis_run_id="analysis_1",
        )


@pytest.mark.parametrize(
    "value, message",
    (
        ('{"2": 5.0, "1": 10.0}', "not canonical JSON"),
        ('{"2":5.0,"1":10.0}', "not canonical JSON"),
        ('{"1":10.0,"1":5.0}', "duplicate component keys"),
        ('{"+1":10.0}', "canonical positive integers"),
        ('{"01":10.0}', "canonical positive integers"),
    ),
)
def test_named_bic_json_must_already_be_canonical_strict_json(
    value: str,
    message: str,
) -> None:
    row = _minimal_row(TRAINING_RESPONSE_CLUSTERS_TABLE)
    row["bic_by_component_count_json"] = value
    with pytest.raises(ValueError, match=message):
        normalize_training_response_row(
            TRAINING_RESPONSE_CLUSTERS_TABLE,
            row,
            analysis_run_id="analysis_1",
        )


def test_named_canonical_bic_json_is_preserved_byte_for_byte() -> None:
    row = _minimal_row(TRAINING_RESPONSE_CLUSTERS_TABLE)
    row["bic_by_component_count_json"] = '{"1":10.0,"2":5.0}'
    normalized = normalize_training_response_row(
        TRAINING_RESPONSE_CLUSTERS_TABLE,
        row,
        analysis_run_id="analysis_1",
    )
    assert normalized["bic_by_component_count_json"] == row[
        "bic_by_component_count_json"
    ]


def test_v2_rejects_alternative_role_derived_columns_and_config() -> None:
    row = _minimal_row(TRAINING_RESPONSE_FEATURES_TABLE)
    row["threatening_pre_p05_distance_mm"] = 1.0
    with pytest.raises(ValueError, match="frozen v2 role vocabulary"):
        normalize_training_response_row(
            TRAINING_RESPONSE_FEATURES_TABLE,
            row,
            analysis_run_id="analysis_1",
        )
    with pytest.raises(ValueError, match="freezes role-derived columns"):
        TrainingResponseConfig(aggressive_role="threatening").validate()


def test_normalization_rejects_null_wrong_type_and_duplicate_primary_keys() -> None:
    row = _minimal_row(TRAINING_RESPONSE_CLASSIFICATION_TABLE)
    row["recording_id"] = None
    with pytest.raises(ValueError, match="required field is null: recording_id"):
        normalize_training_response_row(
            TRAINING_RESPONSE_CLASSIFICATION_TABLE,
            row,
            analysis_run_id="analysis_1",
        )

    row = _minimal_row(TRAINING_RESPONSE_CLASSIFICATION_TABLE)
    row["locomotor_response_metric_count"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        normalize_training_response_row(
            TRAINING_RESPONSE_CLASSIFICATION_TABLE,
            row,
            analysis_run_id="analysis_1",
        )

    normalized = normalize_training_response_rows(
        TRAINING_RESPONSE_FEATURES_TABLE,
        [_minimal_row(TRAINING_RESPONSE_FEATURES_TABLE)],
        analysis_run_id="analysis_1",
    )[0]
    with pytest.raises(ValueError, match="duplicate primary key"):
        validate_training_response_primary_keys(
            TRAINING_RESPONSE_FEATURES_TABLE,
            [normalized, normalized],
        )
    with pytest.raises(ValueError, match="null or empty primary-key"):
        validate_training_response_primary_keys(
            TRAINING_RESPONSE_FEATURES_TABLE,
            [{**normalized, "analysis_run_id": ""}],
        )


def test_empty_batch_has_no_primary_key_sentinel_requirement() -> None:
    assert (
        normalize_training_response_rows(
            TRAINING_RESPONSE_CLUSTERS_TABLE,
            [],
            analysis_run_id="analysis_1",
        )
        == []
    )
