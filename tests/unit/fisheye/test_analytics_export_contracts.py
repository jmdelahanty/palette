from __future__ import annotations

from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    CHASER_CRA_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    canonicalize_export_row,
)


def test_v2_canonicalization_preserves_scientific_values_while_renaming_legacy_vocabulary() -> None:
    source = {
        "recording_id": "recording_1",
        "delta_agg": 1.25,
        "delta_benign": -2.5,
        "specificity_distance": 3.75,
        "benign_color": "#0000ff",
        "object_role": "benign",
    }

    row = canonicalize_export_row(CHASER_CRA_SUMMARY_TABLE, source)

    assert row["delta_agg"] == source["delta_agg"]
    assert row["delta_inert"] == source["delta_benign"]
    assert row["specificity_distance"] == source["specificity_distance"]
    assert row["inert_color"] == source["benign_color"]
    assert row["object_role"] == "inert"
    assert row["export_schema_version"] == EXPORT_SCHEMA_VERSION
    assert row["table_name"] == CHASER_CRA_SUMMARY_TABLE
    assert not any("benign" in key.lower() for key in row)


def test_capability_resolution_requires_complete_canonical_contract_columns() -> None:
    contract = TABLE_CONTRACTS[CHASER_EPOCH_BEHAVIOR_TABLE]
    complete = {CHASER_EPOCH_BEHAVIOR_TABLE: contract.required_columns}

    available = {
        status.capability_id: status for status in resolve_capabilities(complete)
    }["chaser.epoch.behavior_summary"]
    assert available.available is True

    incomplete_columns = tuple(
        column for column in contract.required_columns if column != "mean_speed_mm_s"
    )
    incomplete = {
        status.capability_id: status
        for status in resolve_capabilities(
            {CHASER_EPOCH_BEHAVIOR_TABLE: incomplete_columns}
        )
    }["chaser.epoch.behavior_summary"]
    assert incomplete.available is False
    assert incomplete.missing_columns_by_table == {
        CHASER_EPOCH_BEHAVIOR_TABLE: ("mean_speed_mm_s",)
    }


def test_v1_table_name_is_not_a_registered_or_resolvable_contract() -> None:
    legacy_name = "goodcopbadcop_chaser_epoch_behavior_summary"

    assert legacy_name not in TABLE_CONTRACTS
    statuses = resolve_capabilities({legacy_name: ("recording_id",)})
    behavior = next(
        status
        for status in statuses
        if status.capability_id == "chaser.epoch.behavior_summary"
    )
    assert behavior.available is False
    assert behavior.missing_tables == (CHASER_EPOCH_BEHAVIOR_TABLE,)
