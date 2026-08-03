from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    EXACT_ARROW_SCHEMA_TABLES,
    arrow_contract_envelope,
    exact_arrow_schema,
    validate_arrow_contract_envelope,
    validate_arrow_schema,
)
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_VERSION,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.validation import ExportValidationError, validate_export_run
from fisheye.utils.export_cross_recording_analytics import (
    SourceExportResult,
    _write_table_parts,
    export_sources,
)
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
)
from tests.unit.fisheye.test_export_cross_recording_analytics import (
    _make_source_zarr,
    _write_collection_manifest,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rehash(envelope: dict[str, Any]) -> None:
    for contract in envelope["exact_tables"].values():
        contract["payload_sha256"] = _canonical_sha256(
            {key: value for key, value in contract.items() if key != "payload_sha256"}
        )
    envelope["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in envelope.items() if key != "payload_sha256"}
    )


def _valid_position_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[POSITION_OCCUPANCY_HISTOGRAM_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        elif field.arrow_type == "bool":
            row[field.name] = True
        elif field.arrow_type == "list<string>":
            row[field.name] = ["window", "y_bin", "x_bin"]
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            "recording_id": "recording-1",
            "position_occupancy_path": "analysis/detection_occupancy_runs/run-1",
            "source_refs_json": "{}",
        }
    )
    return row


def _valid_recording_summary_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[RECORDING_SUMMARY_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": RECORDING_SUMMARY_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "a" * 64,
            "stimulus_step_count": 0,
        }
    )
    return row


def test_arrow_contract_envelope_partitions_exact_and_compatibility_tables() -> None:
    envelope = arrow_contract_envelope(
        (POSITION_OCCUPANCY_HISTOGRAM_TABLE, RECORDING_SUMMARY_TABLE)
    )

    assert tuple(envelope["exact_tables"]) == (
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        RECORDING_SUMMARY_TABLE,
    )
    assert envelope["inferred_v2_compatibility_tables"] == []
    assert validate_arrow_contract_envelope(
        envelope,
        (POSITION_OCCUPANCY_HISTOGRAM_TABLE, RECORDING_SUMMARY_TABLE),
    ) == envelope


def test_recording_summary_contract_freezes_exact_field_order_and_nullability() -> None:
    assert EXACT_ARROW_SCHEMA_TABLES == (
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        RECORDING_SUMMARY_TABLE,
    )
    fields = ARROW_TABLE_CONTRACTS[RECORDING_SUMMARY_TABLE].fields
    assert tuple(field.name for field in fields) == (
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
        "stimulus_run",
        "stimulus_response_run",
        "swim_bout_run",
        "stimulus_step_count",
        "protocol_signature_schema",
        "protocol_signature_hash",
        "derived_protocol_hash",
        "protocol_mode_sequence",
        "protocol_duration_sequence_s",
        "protocol_step_count",
        "source_track_kinematics_run",
        "source_track_kinematics_type",
        "source_bout_run",
        "n_fish",
        "n_steps",
        "global_fish_count",
        "total_distance_mm_sum",
        "mean_speed_mm_s_mean",
        "fraction_moving_mean",
        "total_active_s_sum",
        "swim_bout_default_level",
        "swim_bout_default_n_bouts",
        "swim_bout_default_mean_duration_s",
        "swim_bout_default_total_path_length_mm",
        "collection_id",
        "collection_manifest_sha256",
        "collection_manifest_path",
    )
    assert len(fields) == 32
    assert {field.name for field in fields if not field.nullable} == {
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
        "stimulus_step_count",
    }
    assert next(field for field in fields if field.name == "derived_protocol_hash").nullable


@pytest.mark.parametrize(
    "table_name",
    (POSITION_OCCUPANCY_HISTOGRAM_TABLE, RECORDING_SUMMARY_TABLE),
)
@pytest.mark.parametrize(
    "mutation",
    (
        lambda fields: fields.reverse(),
        lambda fields: fields[0].update({"arrow_type": "int64"}),
        lambda fields: fields[0].update({"nullable": True}),
        lambda fields: fields.append(
            {"name": "unexpected", "arrow_type": "string", "nullable": True}
        ),
        lambda fields: fields.pop(),
    ),
    ids=("reordered", "wrong-type", "wrong-nullability", "unexpected", "missing"),
)
def test_rehashed_arrow_contract_tampering_fails_closed(
    table_name: str,
    mutation: Any,
) -> None:
    envelope = arrow_contract_envelope((table_name,))
    fields = envelope["exact_tables"][table_name]["fields"]
    mutation(fields)
    _rehash(envelope)

    with pytest.raises(ValueError, match="differs from installed contracts"):
        validate_arrow_contract_envelope(
            envelope,
            (table_name,),
        )


def test_exact_writer_uses_declared_order_types_nullability_and_digest(tmp_path: Path) -> None:
    table_name = POSITION_OCCUPANCY_HISTOGRAM_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_position_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    expected = exact_arrow_schema(table_name, metadata={})
    assert schema.remove_metadata() == expected.remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_recording_summary_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = RECORDING_SUMMARY_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_recording_summary_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_exact_writer_rejects_unexpected_and_missing_nonnullable_fields(
    tmp_path: Path,
) -> None:
    row = _valid_position_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            rows_by_source=(("source", [row]),),
        )

    row = _valid_position_row()
    del row["hist_count"]
    with pytest.raises(ValueError, match="null/missing non-nullable"):
        _write_table_parts(
            generation_root=tmp_path / "missing",
            table=POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            rows_by_source=(("source", [row]),),
        )


def test_recording_summary_exact_writer_rejects_unexpected_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    row = _valid_recording_summary_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=RECORDING_SUMMARY_TABLE,
            rows_by_source=(("source", [row]),),
        )

    for field_name in (
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
        "stimulus_step_count",
    ):
        row = _valid_recording_summary_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=RECORDING_SUMMARY_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_recording_summary_zero_rows_publish_no_parts_but_retain_exact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={RECORDING_SUMMARY_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-recording-summary",
        tables=(RECORDING_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {RECORDING_SUMMARY_TABLE: 0}
    assert manifest["part_files_by_table"] == {RECORDING_SUMMARY_TABLE: []}
    assert manifest["publication"]["parts_by_table"] == {
        RECORDING_SUMMARY_TABLE: []
    }
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        RECORDING_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-recording-summary")["status"] == "valid"


def test_real_detection_occupancy_export_uses_exact_arrow_contract(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="occupancy-arrow",
        tables=(POSITION_OCCUPANCY_HISTOGRAM_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"][POSITION_OCCUPANCY_HISTOGRAM_TABLE] > 0
    part = root / manifest["part_files_by_table"][POSITION_OCCUPANCY_HISTOGRAM_TABLE][0]
    validate_arrow_schema(
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        pq.ParquetFile(part).schema_arrow,
    )
    assert validate_export_run(root, "occupancy-arrow")["status"] == "valid"


def test_real_recording_summary_export_uses_exact_schema_and_collection_fields(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "recording_a_analysis.zarr")
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path, source)
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="recording-summary-arrow",
        tables=(RECORDING_SUMMARY_TABLE,),
        jobs=1,
        collection_manifest_path=collection_path,
    )

    assert manifest["row_counts_by_table"] == {RECORDING_SUMMARY_TABLE: 1}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        RECORDING_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    part = root / manifest["part_files_by_table"][RECORDING_SUMMARY_TABLE][0]
    schema = pq.ParquetFile(part).schema_arrow
    validate_arrow_schema(RECORDING_SUMMARY_TABLE, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        RECORDING_SUMMARY_TABLE,
        metadata={},
    ).remove_metadata()
    row = pq.read_table(part).to_pylist()[0]
    assert row["recording_id"] == "recording_a"
    assert row["stimulus_step_count"] == 2
    assert row["collection_id"] == collection["collection_id"]
    assert row["collection_manifest_sha256"] == collection["manifest_sha256"]
    assert validate_export_run(root, "recording-summary-arrow")["status"] == "valid"


def test_manifest_selected_reader_rejects_rehashed_wrong_physical_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_name = POSITION_OCCUPANCY_HISTOGRAM_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_position_row()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr", source
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="exact-arrow",
        tables=(table_name,),
        jobs=1,
    )
    assert validate_export_run(root, "exact-arrow")["status"] == "valid"

    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.read_table(part)
    column_index = original.schema.get_field_index("hist_count")
    columns = list(original.columns)
    columns[column_index] = pa.array([1.0], type=pa.float64())
    wrong_schema = original.schema.set(
        column_index,
        pa.field("hist_count", pa.float64(), nullable=False),
    )
    pq.write_table(pa.Table.from_arrays(columns, schema=wrong_schema), part)

    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="physical Arrow fields"):
        validate_export_run(root, "exact-arrow")


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "wrong_type", "wrong_nullability", "unexpected", "missing", "metadata"),
)
def test_recording_summary_manifest_reader_rejects_rehashed_physical_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    table_name = RECORDING_SUMMARY_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_recording_summary_row()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id=f"recording-summary-{mutation}",
        tables=(table_name,),
        jobs=1,
    )
    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.ParquetFile(part).read()
    arrays = list(original.columns)
    fields = list(original.schema)
    metadata = dict(original.schema.metadata or {})

    if mutation == "reordered":
        arrays[0], arrays[1] = arrays[1], arrays[0]
        fields[0], fields[1] = fields[1], fields[0]
    elif mutation == "wrong_type":
        index = original.schema.get_field_index("stimulus_step_count")
        arrays[index] = pa.array([0.0], type=pa.float64())
        fields[index] = pa.field("stimulus_step_count", pa.float64(), nullable=False)
    elif mutation == "wrong_nullability":
        index = original.schema.get_field_index("recording_id")
        fields[index] = pa.field("recording_id", pa.string(), nullable=True)
    elif mutation == "unexpected":
        arrays.append(pa.array(["surprise"], type=pa.string()))
        fields.append(pa.field("unexpected", pa.string(), nullable=False))
    elif mutation == "missing":
        index = original.schema.get_field_index("source_lineage_hash")
        del arrays[index]
        del fields[index]
    elif mutation == "metadata":
        metadata[b"palette.arrow_schema_sha256"] = b"0" * 64
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)

    wrong_schema = pa.schema(fields, metadata=metadata)
    pq.write_table(pa.Table.from_arrays(arrays, schema=wrong_schema), part)

    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExportValidationError,
        match="physical Arrow fields|footer contract metadata",
    ):
        validate_export_run(root, f"recording-summary-{mutation}")
