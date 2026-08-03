from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    EXPORT_SCHEMA_VERSION,
    EXPORT_SCHEMA_ID,
    RECORDING_SUMMARY_TABLE,
    STIMULUS_STEP_SUMMARY_TABLE,
    TABLE_CONTRACTS,
    canonicalize_export_row,
    contract_snapshot,
)
from fisheye.analytics_exports.validation import ExportValidationError, validate_export_run


def test_v2_canonicalization_preserves_scientific_values_while_renaming_legacy_vocabulary() -> None:
    source = {
        "recording_id": "recording_1",
        "delta_agg": 1.25,
        "delta_benign": -2.5,
        "specificity_distance": 3.75,
        "benign_color": "#0000ff",
        "object_role": "benign",
    }

    row = canonicalize_export_row(CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE, source)

    assert row["delta_agg"] == source["delta_agg"]
    assert row["delta_inert"] == source["delta_benign"]
    assert row["specificity_distance"] == source["specificity_distance"]
    assert row["inert_color"] == source["benign_color"]
    assert row["object_role"] == "inert"
    assert row["export_schema_version"] == EXPORT_SCHEMA_VERSION
    assert row["table_name"] == CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE
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


def test_stimulus_step_summary_contract_key_is_unique_for_multiple_fish_per_step() -> None:
    contract = TABLE_CONTRACTS[STIMULUS_STEP_SUMMARY_TABLE]
    rows = (
        {"recording_id": "recording-1", "fish_id": 7, "step_index": 3},
        {"recording_id": "recording-1", "fish_id": 8, "step_index": 3},
    )

    keys = {
        tuple(row[field] for field in contract.primary_key)
        for row in rows
    }

    assert contract.grain == "recording_x_fish_x_stimulus_step_summary"
    assert contract.primary_key == ("recording_id", "fish_id", "step_index")
    assert keys == {("recording-1", 7, 3), ("recording-1", 8, 3)}


def _write_valid_export(root: Path, run_id: str) -> Path:
    table_name = RECORDING_SUMMARY_TABLE
    part = root / "v1" / table_name / f"export_run_id={run_id}" / "part-00000.parquet"
    part.parent.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "table_name": table_name,
                "recording_id": "recording_1",
            }
        ]
    )
    table = table.replace_schema_metadata(
        {
            b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
            b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(),
            b"palette.table_contract": json.dumps(
                TABLE_CONTRACTS[table_name].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        }
    )
    pq.write_table(table, part)
    manifest = root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "export_run_id": run_id,
                "schema_id": EXPORT_SCHEMA_ID,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "tables_requested": [table_name],
                "table_contracts": contract_snapshot([table_name]),
                "row_counts_by_table": {table_name: 1},
                "part_files_by_table": {table_name: [str(part)]},
                "capabilities": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_export_validator_checks_every_part_and_manifest_row_count(tmp_path: Path) -> None:
    root = tmp_path / "analytics"
    manifest = _write_valid_export(root, "valid_export")

    with pytest.raises(ExportValidationError, match="explicit legacy compatibility"):
        validate_export_run(root, "valid_export")
    report = validate_export_run(root, "valid_export", allow_legacy_layout=True)

    assert report["status"] == "valid"
    assert report["part_count"] == 1
    assert report["row_count"] == 1

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["row_counts_by_table"][RECORDING_SUMMARY_TABLE] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExportValidationError, match="row count 1 does not match manifest 2"):
        validate_export_run(root, "valid_export", allow_legacy_layout=True)
