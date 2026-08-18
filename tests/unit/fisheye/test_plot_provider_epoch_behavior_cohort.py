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
    _distribution_data,
    plot_provider_epoch_behavior_cohort_parquet,
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


def _manifest(*, recording_count: int = 3, bout_count: int | None = None) -> dict[str, object]:
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
    row_counts = {
        TABLE_BOUTS: recording_count if bout_count is None else bout_count,
        TABLE_FISH: recording_count * 3,
    }
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


def _distribution_tables() -> tuple[pa.Table, pa.Table, int]:
    bouts: list[dict[str, object]] = []
    fish_rows: list[dict[str, object]] = []
    sessions = (
        (
            "recording-a",
            "fish-1",
            (
                ((1.0, 2.0), (3.0, 6.0), (0.0, 1.0)),
                ((2.0, 4.0), (1.0, -1.0)),
                ((1.0, 1.0),),
            ),
        ),
        (
            "recording-b",
            "fish-1",
            (
                ((5.0, 10.0), (7.0, 14.0)),
                ((4.0, 12.0),),
                ((2.0, 4.0),),
            ),
        ),
        (
            "recording-c",
            "fish-2",
            (
                ((10.0, 20.0),),
                ((8.0, 8.0),),
                ((float("nan"), 3.0),),
            ),
        ),
    )
    bout_source_row = 0
    for recording_id, subject_id, epoch_rows in sessions:
        for epoch_id, values in enumerate(epoch_rows):
            fish_rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "track_id": 0,
                    "epoch_id": epoch_id,
                    "epoch_index": epoch_id,
                    "epoch_label": EXPECTED_EPOCH_LABELS[epoch_id],
                    "mean_speed_mm_s": 1.0,
                    "mean_bout_duration_s": 1.0,
                    "bout_rate_per_min": 1.0,
                }
            )
            for duration, path_length in values:
                bouts.append(
                    {
                        "recording_id": recording_id,
                        "subject_id": subject_id,
                        "track_id": 0,
                        "epoch_id": epoch_id,
                        "epoch_index": epoch_id,
                        "epoch_label": EXPECTED_EPOCH_LABELS[epoch_id],
                        "bout_source_row": bout_source_row,
                        "bout_duration_s": duration,
                        "bout_path_length_mm": path_length,
                    }
                )
                bout_source_row += 1
    return _make_table(TABLE_BOUTS, bouts), _make_table(TABLE_FISH, fish_rows), len(bouts)


def _recording_mode_fixture(tmp_path: Path) -> tuple[pa.Table, pa.Table, dict[str, object], Path, str]:
    bouts: list[dict[str, object]] = []
    fish_rows: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    collisions: list[dict[str, object]] = []
    bout_source_row = 0
    for index in range(16):
        recording_id = f"recording-{index:02d}"
        source_subject_id = f"source-subject-{index // 2:02d}"
        entries.append(
            {
                "recording_id": recording_id,
                "analysis_zarr": f"/immutable/{recording_id}.zarr",
                "summary_run": f"summary-{index:02d}",
                "track_id": 0,
                "subject_id": source_subject_id,
            }
        )
        if index % 2 == 1:
            collisions.append(
                {
                    "source_subject_id": source_subject_id,
                    "recording_ids": [f"recording-{index - 1:02d}", recording_id],
                }
            )
        for epoch_id, epoch_label in enumerate(EXPECTED_EPOCH_LABELS):
            fish_rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": source_subject_id,
                    "track_id": 0,
                    "epoch_id": epoch_id,
                    "epoch_index": epoch_id,
                    "epoch_label": epoch_label,
                    "mean_speed_mm_s": 1.0,
                    "mean_bout_duration_s": 1.0,
                    "bout_rate_per_min": 1.0,
                }
            )
            bouts.append(
                {
                    "recording_id": recording_id,
                    "subject_id": source_subject_id,
                    "track_id": 0,
                    "epoch_id": epoch_id,
                    "epoch_index": epoch_id,
                    "epoch_label": epoch_label,
                    "bout_source_row": bout_source_row,
                    "bout_duration_s": 1.0,
                    "bout_path_length_mm": float(epoch_id + 1),
                }
            )
            bout_source_row += 1

    input_manifest: dict[str, object] = {
        "schema_id": "palette.provider_epoch_behavior_cohort_input",
        "schema_version": 1,
        "cohort_id": "goodbatbadbat-talk",
        "metric_disposition": "linear_only",
        "metric_disposition_reason": "test",
        "entries": entries,
    }
    input_manifest["manifest_payload_sha256"] = canonical_json_sha256(input_manifest)
    input_path = tmp_path / "input_manifest.json"
    input_path.write_text(json.dumps(input_manifest), encoding="utf-8")
    publication_manifest = _manifest(recording_count=16, bout_count=len(bouts))
    publication_manifest["input_manifest_path"] = str(input_path)
    publication_manifest["input_manifest_sha256"] = canonical_json_sha256(input_manifest)
    publication_manifest["manifest_payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in publication_manifest.items() if key != "manifest_payload_sha256"}
    )
    decision: dict[str, object] = {
        "schema_id": "palette.cohort_analysis_unit_decision",
        "schema_version": 1,
        "decision_id": "decision-recording-mode-test",
        "cohort_id": "goodbatbadbat-talk",
        "source_manifest_sha256": publication_manifest["input_manifest_sha256"],
        "source_manifest_payload_sha256": input_manifest["manifest_payload_sha256"],
        "analysis_unit": "recording_id",
        "policy_id": "operator_asserted_distinct_animal_per_recording_v1",
        "operator_assertion": "each_recording_contains_a_distinct_animal",
        "operator_identity": "operator@example.org",
        "reviewed_at_utc": "2026-08-18T12:00:00Z",
        "canonical_subject_identity_corrected": False,
        "reason_code": "acquisition_subject_id_reuse",
        "recording_count": 16,
        "duplicate_source_subject_id_count": 8,
        "affected_recording_count": 16,
        "collisions": collisions,
        "source_manifest_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    decision["decision_payload_sha256"] = canonical_json_sha256(decision)
    decision_path = tmp_path / "analysis_unit_decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return (
        _make_table(TABLE_BOUTS, bouts),
        _make_table(TABLE_FISH, fish_rows),
        publication_manifest,
        decision_path,
        str(publication_manifest["input_manifest_sha256"]),
    )


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


def test_distribution_aggregation_and_speed_validity_are_explicit() -> None:
    bouts, fish, bout_count = _distribution_tables()
    data = validate_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=_manifest(bout_count=bout_count),
    )
    distributions = _distribution_data(data)

    duration = distributions.metrics["bout_duration_s"]
    assert duration["total_bout_count_by_epoch"] == [6, 4, 3]
    assert duration["valid_bout_count_by_epoch"] == [5, 4, 2]
    assert duration["dropped_bout_count_by_epoch"] == [1, 0, 1]
    assert duration["dropped_reason_counts_by_epoch"] == [
        {"zero_or_nonpositive": 1},
        {},
        {"nonfinite": 1},
    ]
    path_length = distributions.metrics["bout_path_length_mm"]
    assert path_length["valid_bout_count_by_epoch"] == [6, 3, 3]
    assert path_length["dropped_reason_counts_by_epoch"][1] == {"negative": 1}
    speed = distributions.metrics["mean_bout_speed_mm_s"]
    assert speed["valid_bout_count_by_epoch"] == [5, 3, 2]
    assert speed["dropped_reason_counts_by_epoch"] == [
        {"duration_zero_or_nonpositive": 1},
        {"path_length_negative": 1},
        {"duration_nonfinite": 1},
    ]
    assert distributions.pooled_values_by_metric_epoch["mean_bout_speed_mm_s"][0].tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
    # fish-1 has two sessions.  Session medians are combined equally (4.0),
    # rather than weighting the subject by its pooled bout count.
    assert distributions.subject_values_by_metric_epoch["bout_duration_s"][0].tolist() == [4.0, 10.0]
    assert distributions.metrics["bout_duration_s"]["valid_session_median_count_by_epoch"] == [3, 3, 2]
    assert distributions.metrics["bout_duration_s"]["valid_subject_count_by_epoch"] == [2, 2, 1]


def test_distribution_figures_and_receipt_are_deterministic(tmp_path: Path) -> None:
    bouts, fish, bout_count = _distribution_tables()
    manifest = _manifest(bout_count=bout_count)
    first = plot_provider_epoch_behavior_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=manifest,
        output_dir=tmp_path / "first",
        prefix="distribution",
    )
    second = plot_provider_epoch_behavior_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=manifest,
        output_dir=tmp_path / "second",
        prefix="distribution",
    )

    first_paths = sorted(Path(path) for path in first["figure_paths"])
    second_paths = sorted(Path(path) for path in second["figure_paths"])
    assert [path.name for path in first_paths] == [path.name for path in second_paths]
    for first_path, second_path in zip(first_paths, second_paths):
        assert first_path.read_bytes() == second_path.read_bytes()
    assert {
        "distribution.pooled_bout_ecdf.png",
        "distribution.pooled_bout_ecdf.svg",
        "distribution.subject_balanced_bout_distributions.png",
        "distribution.subject_balanced_bout_distributions.svg",
    } <= {path.name for path in first_paths}
    receipt = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["distribution_figures"]["pooled_bout_ecdf"]["x_scale"] == "log10"
    assert receipt["distribution_figures"]["pooled_bout_ecdf"]["clipping"] == "none"
    assert receipt["distribution_figures"]["subject_balanced_bout_distributions"]["y_scale"] == "linear"
    assert receipt["distribution_metrics"]["bout_path_length_mm"]["valid_bout_count_by_epoch"] == [6, 3, 3]
    assert receipt["distribution_metrics"]["mean_bout_speed_mm_s"]["valid_bout_count_by_epoch"] == [5, 3, 2]
    figure_receipt_names = {entry["path"] for entry in receipt["figures"]}
    assert {
        "distribution.pooled_bout_ecdf.png",
        "distribution.subject_balanced_bout_distributions.png",
    } <= figure_receipt_names
    for entry in receipt["figures"]:
        assert len(entry["sha256"]) == 64


def test_exact_parquet_plotter_reads_physical_schema_without_hive_fields(tmp_path: Path) -> None:
    bouts, fish, bout_count = _distribution_tables()
    output_root = tmp_path / "cohort"
    manifest_dir = output_root / "v2" / "manifests"
    generation = output_root / "v2" / ".generations" / "analysis_run_id=talk-run" / "generation=generation-a"
    manifest_dir.mkdir(parents=True)
    generation.mkdir(parents=True)
    manifest = _manifest(bout_count=bout_count)
    parts = manifest["publication"]["parts_by_table"]
    assert isinstance(parts, dict)
    for table_name, table in ((TABLE_BOUTS, bouts), (TABLE_FISH, fish)):
        path = generation / "tables" / table_name / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)
        entry = parts[table_name][0]
        assert isinstance(entry, dict)
        entry["path"] = str(path.relative_to(output_root))
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["size_bytes"] = path.stat().st_size
        entry["row_count"] = table.num_rows
        manifest["part_files_by_table"][table_name] = [entry["path"]]
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_payload_sha256"
    }
    manifest["manifest_payload_sha256"] = canonical_json_sha256(unsigned_manifest)
    manifest_path = manifest_dir / "talk-run.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = plot_provider_epoch_behavior_cohort_parquet(
        bouts_parquet=generation / "tables" / TABLE_BOUTS / "part-00000.parquet",
        fish_parquet=generation / "tables" / TABLE_FISH / "part-00000.parquet",
        manifest_path=manifest_path,
        output_dir=tmp_path / "plots",
        prefix="physical",
    )
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert "analysis_run_id" not in receipt["source_tables"][TABLE_BOUTS]
    assert Path(result["receipt_path"]).exists()


def test_recording_analysis_units_bind_decision_and_use_one_unit_per_recording(
    tmp_path: Path,
) -> None:
    bouts, fish, manifest, decision_path, canonical_input_sha = _recording_mode_fixture(tmp_path)
    data = validate_cohort_tables(bouts_table=bouts, fish_table=fish, manifest=manifest)
    result = plot_provider_epoch_behavior_cohort_tables(
        bouts_table=bouts,
        fish_table=fish,
        manifest=manifest,
        output_dir=tmp_path / "recording-plots",
        prefix="recording",
        analysis_unit_mode="recording",
        analysis_unit_decision_path=decision_path,
    )
    assert data.n_recording_animal_sessions == 16
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["analysis_unit_mode"] == "recording"
    assert receipt["analysis_unit_label"] == "recording×animal unit"
    assert receipt["analysis_unit_count"] == 16
    assert receipt["grouping_unit"] == "recording_id"
    assert receipt["grouped_estimate_level"] == "recording_id"
    assert receipt["repeated_session_aggregation"] == "not_applicable_recording_id_is_analysis_unit"
    assert receipt["distribution_metrics"]["bout_duration_s"]["valid_analysis_unit_count_by_epoch"] == [16, 16, 16]
    assert receipt["distribution_metrics"]["bout_duration_s"]["valid_subject_count_by_epoch"] is None
    assert receipt["canonical_subject_identity_corrected"] is False
    assert receipt["duplicate_source_subject_id_count"] == 8
    assert receipt["affected_recording_count"] == 16
    decision_binding = receipt["analysis_unit_decision"]
    assert decision_binding["payload"]["source_manifest_sha256"] == canonical_input_sha
    assert decision_binding["sha256"] == hashlib.sha256(decision_path.read_bytes()).hexdigest()
    assert Path(tmp_path / "recording-plots" / "recording.grouped_bout_rate_per_min.png").exists()
    assert "recording×animal units" in (
        Path(tmp_path / "recording-plots" / "recording.subject_balanced_bout_distributions.svg").read_text(
            encoding="utf-8"
        )
    )


def test_recording_analysis_units_fail_closed_without_decision_evidence(tmp_path: Path) -> None:
    bouts, fish = _tables()
    with pytest.raises(ProviderEpochBehaviorPlotError, match="requires an immutable analysis-unit decision"):
        plot_provider_epoch_behavior_cohort_tables(
            bouts_table=bouts,
            fish_table=fish,
            manifest=_manifest(),
            output_dir=tmp_path / "recording-plots",
            analysis_unit_mode="recording",
        )


def test_recording_decision_distinguishes_canonical_and_raw_manifest_identity(
    tmp_path: Path,
) -> None:
    bouts, fish, manifest, decision_path, canonical_input_sha = _recording_mode_fixture(tmp_path)
    raw_input_sha = hashlib.sha256(
        Path(manifest["input_manifest_path"]).read_bytes()
    ).hexdigest()
    assert canonical_input_sha != raw_input_sha
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["input_manifest_sha256"] = raw_input_sha
    bad_manifest["manifest_payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in bad_manifest.items() if key != "manifest_payload_sha256"}
    )
    with pytest.raises(ProviderEpochBehaviorPlotError, match="canonical input manifest identity"):
        plot_provider_epoch_behavior_cohort_tables(
            bouts_table=bouts,
            fish_table=fish,
            manifest=bad_manifest,
            output_dir=tmp_path / "bad-identity",
            analysis_unit_mode="recording",
            analysis_unit_decision_path=decision_path,
        )


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
