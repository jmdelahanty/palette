from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contract_core import contract_envelope, exact_schema
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.export_provider_epoch_behavior_cohort import (
    ARROW_ENVELOPE_SCHEMA_ID,
    ARROW_ENVELOPE_SCHEMA_VERSION,
    TABLE_BOUTS,
    TABLE_FISH,
    TABLE_NAMES,
    table_contracts_for_disposition,
)
from fisheye.utils.plot_provider_epoch_behavior_cohort import (
    EXPECTED_EPOCH_LABELS,
    NEUTRAL_EPOCH_COLORS,
    ProviderEpochBehaviorPlotError,
    plot_provider_epoch_behavior_cohort_tables,
    validate_cohort_tables,
)


def _default_value(arrow_type: str, nullable: bool) -> object:
    if nullable:
        return None
    if arrow_type.startswith("float"):
        return 0.0
    if arrow_type.startswith("int") or arrow_type.startswith("uint"):
        return 0
    if arrow_type == "string":
        return "value"
    raise AssertionError(arrow_type)


def _make_table(table_name: str, rows: list[dict[str, object]]) -> pa.Table:
    contract = table_contracts_for_disposition("linear_only")[table_name]
    normalized = [
        {
            field.name: row.get(field.name, _default_value(field.arrow_type, field.nullable))
            for field in contract.fields
        }
        for row in rows
    ]
    return pa.Table.from_pylist(
        normalized,
        schema=exact_schema(contract, metadata={
            b"palette.export_schema_id": b"palette.provider_epoch_behavior_cohort",
            b"palette.export_schema_version": b"1",
            b"palette.selector_eligible": b"false",
            b"palette.table_name": table_name.encode("utf-8"),
        }),
    )


def _manifest(*, recording_count: int = 3) -> dict[str, object]:
    contracts = table_contracts_for_disposition("linear_only")
    envelope = contract_envelope(
        TABLE_NAMES,
        known_table_names=TABLE_NAMES,
        contracts=contracts,
        schema_id=ARROW_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
    )
    analysis_run_id = "talk-run"
    generation_id = "generation-a"
    generation_path = f"v2/.generations/analysis_run_id={analysis_run_id}/generation={generation_id}"
    row_counts = {TABLE_BOUTS: recording_count, TABLE_FISH: recording_count * 3}
    parts = {
        table: [
            {
                "path": f"{generation_path}/tables/{table}/part-00000.parquet",
                "sha256": hashlib.sha256(table.encode("ascii")).hexdigest(),
                "size_bytes": 100,
                "row_count": row_counts[table],
            }
        ]
        for table in TABLE_NAMES
    }
    publication = {
        "schema_id": "palette.derived_analytics.publication",
        "schema_version": 1,
        "state": "complete",
        "selector_eligible": False,
        "intended_use": "analysis",
        "generation_id": generation_id,
        "generation_path": generation_path,
        "parts_by_table": parts,
    }
    payload: dict[str, object] = {
        "export_schema_id": "palette.provider_epoch_behavior_cohort",
        "export_schema_version": 1,
        "cohort_id": "goodbatbadbat-talk",
        "analysis_run_id": analysis_run_id,
        "metric_disposition": "linear_only",
        "metric_disposition_reason": "heading caches are not used for this linear-motion talk cohort",
        "excluded_metrics": ["bout_net_heading_change_deg"],
        "recording_count": recording_count,
        "selector_eligible": False,
        "source_lineage": [],
        "output_tables": list(TABLE_NAMES),
        "row_counts_by_table": row_counts,
        "part_files_by_table": {
            table: [parts[table][0]["path"]] for table in TABLE_NAMES
        },
        "primary_keys_by_table": {
            table: list(contracts[table].primary_key) for table in TABLE_NAMES
        },
        "arrow_schema_contracts": envelope,
        "publication": publication,
    }
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _tables() -> tuple[pa.Table, pa.Table]:
    fish_rows: list[dict[str, object]] = []
    bout_rows: list[dict[str, object]] = []
    for recording_index, recording_id in enumerate(("recording-a", "recording-b", "recording-c")):
        subject_id = "fish-1" if recording_index < 2 else "fish-2"
        speed_base = (1.0, 2.0, 100.0)[recording_index]
        rate_base = (1.0, 2.0, 100.0)[recording_index]
        for epoch_id, epoch_label in enumerate(EXPECTED_EPOCH_LABELS):
            fish_rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "track_id": 0,
                    "epoch_id": epoch_id,
                    "epoch_index": epoch_id,
                    "epoch_label": epoch_label,
                    "mean_speed_mm_s": speed_base + epoch_id,
                    "mean_bout_duration_s": float(recording_index + 1) / (epoch_id + 1),
                    "bout_rate_per_min": rate_base * (epoch_id + 1),
                }
            )
            if epoch_id == 0:
                bout_rows.append(
                    {
                        "recording_id": recording_id,
                        "subject_id": subject_id,
                        "track_id": 0,
                        "epoch_id": epoch_id,
                        "epoch_index": epoch_id,
                        "epoch_label": epoch_label,
                        "bout_source_row": recording_index,
                    }
                )
    return _make_table(TABLE_BOUTS, bout_rows), _make_table(TABLE_FISH, fish_rows)


def test_validation_preserves_repeated_recordings_as_distinct_units() -> None:
    bouts, fish = _tables()
    validated = validate_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=_manifest(),
    )

    assert len(validated.units) == 3
    assert validated.units[0].subject_id == validated.units[1].subject_id == "fish-1"
    assert validated.units[2].subject_id == "fish-2"
    assert validated.units[0].unit_id != validated.units[1].unit_id
    assert validated.units[0].values_by_metric["bout_rate_per_min"] == (1.0, 2.0, 3.0)


def test_plot_outputs_are_deterministic_and_semantically_neutral(tmp_path: Path) -> None:
    bouts, fish = _tables()
    manifest = _manifest()
    first = plot_provider_epoch_behavior_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=manifest,
        output_dir=tmp_path / "first",
    )
    second = plot_provider_epoch_behavior_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=manifest,
        output_dir=tmp_path / "second",
    )

    first_paths = sorted(Path(path) for path in first["figure_paths"])
    second_paths = sorted(Path(path) for path in second["figure_paths"])
    assert [path.suffix for path in first_paths] == [path.suffix for path in second_paths]
    for first_path, second_path in zip(first_paths, second_paths):
        assert first_path.read_bytes() == second_path.read_bytes()
    assert (tmp_path / "first" / "provider_epoch_behavior_cohort.individual_bout_rate.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "first" / "provider_epoch_behavior_cohort.individual_bout_rate.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")

    receipt = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["metric_disposition"] == "linear_only"
    assert receipt["expected_epoch_labels"] == list(EXPECTED_EPOCH_LABELS)
    assert receipt["n_recordings"] == 3
    assert receipt["n_subjects"] == 2
    assert receipt["recording_animal_unit_count"] == 3
    assert receipt["epoch_colors"] == NEUTRAL_EPOCH_COLORS
    assert receipt["metrics"]["mean_speed_mm_s"]["mean_by_epoch"] == [50.75, 51.75, 52.75]
    assert receipt["session_weighting"] == "equal"
    assert receipt["session_counts_by_subject"] == {"fish-1": 2, "fish-2": 1}
    assert receipt["metrics"]["bout_rate_per_min"]["finite_subject_counts_by_epoch"] == [2, 2, 2]


def test_validation_rejects_non_linear_disposition_and_wrong_epoch_label() -> None:
    bouts, fish = _tables()
    full_manifest = copy.deepcopy(_manifest())
    full_manifest["metric_disposition"] = "full"
    with pytest.raises(ProviderEpochBehaviorPlotError, match="linear_only"):
        validate_cohort_tables(bouts_table=bouts, fish_table=fish, manifest=full_manifest)

    bad_rows = fish.to_pylist()
    bad_rows[1]["epoch_label"] = "other_event"
    bad_fish = _make_table(TABLE_FISH, bad_rows)
    with pytest.raises(ProviderEpochBehaviorPlotError, match="ordered pre/training/post"):
        validate_cohort_tables(bouts_table=bouts, fish_table=bad_fish, manifest=_manifest())


def test_grouped_plot_rejects_missing_subject_identity(tmp_path: Path) -> None:
    bouts, fish = _tables()
    rows = fish.to_pylist()
    for row in rows[:3]:
        row["subject_id"] = None
    missing_subject_fish = _make_table(TABLE_FISH, rows)

    with pytest.raises(ProviderEpochBehaviorPlotError, match="subject_id"):
        plot_provider_epoch_behavior_cohort_tables(
            bouts_table=bouts,
            fish_table=missing_subject_fish,
            manifest=_manifest(),
            output_dir=tmp_path / "plots",
        )


def test_exact_file_reader_does_not_inject_hive_partition_columns(
    tmp_path: Path,
) -> None:
    bouts, fish = _tables()
    generation = (
        tmp_path
        / "analysis_run_id=talk-run"
        / "generation=generation-a"
    )
    generation.mkdir(parents=True)
    bouts_path = generation / "bouts.parquet"
    fish_path = generation / "fish.parquet"
    pq.write_table(bouts, bouts_path)
    pq.write_table(fish, fish_path)
    validated = validate_cohort_tables(
        bouts_table=pq.ParquetFile(bouts_path).read(),
        fish_table=pq.ParquetFile(fish_path).read(),
        manifest=_manifest(),
    )
    assert validated.bouts_table.schema.names == bouts.schema.names
    assert "analysis_run_id" not in validated.bouts_table.schema.names
