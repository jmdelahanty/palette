from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    EXACT_ARROW_SCHEMA_TABLES,
    arrow_contract_envelope,
    exact_arrow_schema,
    validate_arrow_contract_envelope,
    validate_arrow_schema,
)
from fisheye.analytics_exports.contracts import (
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    EXPORT_SCHEMA_VERSION,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.validation import ExportValidationError, validate_export_run
from fisheye.shared.zarr.columnar import write_columnar_dataset
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


def _valid_baseline_summary_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": BASELINE_BEHAVIOR_SUMMARY_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "b" * 64,
            "source_refs_json": "{}",
        }
    )
    return row


def test_arrow_contract_envelope_partitions_exact_and_compatibility_tables() -> None:
    envelope = arrow_contract_envelope(
        (
            POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            RECORDING_SUMMARY_TABLE,
            BASELINE_BEHAVIOR_SUMMARY_TABLE,
        )
    )

    assert tuple(envelope["exact_tables"]) == (
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        RECORDING_SUMMARY_TABLE,
        BASELINE_BEHAVIOR_SUMMARY_TABLE,
    )
    assert envelope["inferred_v2_compatibility_tables"] == []
    assert (
        validate_arrow_contract_envelope(
            envelope,
            (
                POSITION_OCCUPANCY_HISTOGRAM_TABLE,
                RECORDING_SUMMARY_TABLE,
                BASELINE_BEHAVIOR_SUMMARY_TABLE,
            ),
        )
        == envelope
    )


def test_recording_summary_contract_freezes_exact_field_order_and_nullability() -> None:
    assert EXACT_ARROW_SCHEMA_TABLES == (
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        RECORDING_SUMMARY_TABLE,
        BASELINE_BEHAVIOR_SUMMARY_TABLE,
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


def test_baseline_summary_contract_freezes_all_95_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("chaser_distance_run", "string", False),
        ("chaser_distance_path", "string", False),
        ("chaser_distance_schema_id", "string", True),
        ("chaser_distance_schema_version", "int64", True),
        ("chaser_distance_method", "string", True),
        ("chaser_distance_method_version", "string", True),
        ("source_detection_path", "string", True),
        ("source_detection_kind", "string", True),
        ("source_stimulus_run", "string", True),
        ("source_stimulus_path", "string", True),
        ("source_stimulus_epoch_run", "string", True),
        ("source_stimulus_epoch_path", "string", True),
        ("source_refs_json", "string", False),
        ("coordinate_frame", "string", False),
        ("coordinate_origin", "string", False),
        ("fps", "float64", True),
        ("total_frames", "int64", True),
        ("pixels_per_mm_projector", "float64", False),
        ("source_chaser_distance_run", "string", False),
        ("source_chaser_distance_path", "string", False),
        ("source_epoch_behavior_component", "string", False),
        ("source_epoch_behavior_path", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_kinematics_track_path", "string", False),
        ("source_speed_level", "string", False),
        ("source_swim_bout_run", "string", True),
        ("source_swim_bout_path", "string", True),
        ("track_id", "int64", False),
        ("arena_center_x_px", "float64", False),
        ("arena_center_y_px", "float64", False),
        ("arena_radius_px", "float64", False),
        ("baseline_method", "string", False),
        ("baseline_method_version", "string", False),
        ("baseline_window_id", "int64", False),
        ("baseline_window_label", "string", False),
        ("start_frame", "int64", False),
        ("end_frame", "int64", False),
        ("start_time_s", "float64", False),
        ("end_time_s", "float64", False),
        ("duration_s", "float64", False),
        ("total_frame_count", "int64", False),
        ("valid_frame_count", "int64", False),
        ("missing_frame_count", "int64", False),
        ("tracking_dropout_fraction", "float64", True),
        ("speed_sample_count", "int64", False),
        ("mean_speed_mm_s", "float64", True),
        ("median_speed_mm_s", "float64", True),
        ("p95_speed_mm_s", "float64", True),
        ("max_speed_mm_s", "float64", True),
        ("total_path_mm", "float64", True),
        ("bout_count", "int64", False),
        ("bout_rate_per_min", "float64", True),
        ("arena_radius_mm", "float64", False),
        ("wall_band_mm", "float64", False),
        ("expected_uniform_wall_fraction", "float64", False),
        ("experimental_area_geometry_type", "string", False),
        ("boundary_distance_method", "string", False),
        ("wall_fraction_denominator", "string", False),
        ("wall_frame_count", "int64", False),
        ("wall_fraction", "float64", True),
        ("mean_distance_from_arena_center_mm", "float64", True),
        ("median_distance_from_arena_center_mm", "float64", True),
        ("p95_distance_from_arena_center_mm", "float64", True),
        ("mean_distance_to_arena_boundary_mm", "float64", True),
        ("median_distance_to_arena_boundary_mm", "float64", True),
        ("p95_distance_to_arena_boundary_mm", "float64", True),
        ("mean_center_distance_norm", "float64", True),
        ("median_center_distance_norm", "float64", True),
        ("x_axis_direction", "string", False),
        ("y_axis_direction", "string", False),
        ("spatial_grid_size", "int64", False),
        ("spatial_valid_sample_count", "int64", False),
        ("spatial_visited_cell_count", "int64", False),
        ("spatial_entropy_normalized", "float64", True),
        ("spatial_max_cell_fraction", "float64", True),
        ("quadrant_entropy_normalized", "float64", True),
        ("quadrant_max_fraction", "float64", True),
        ("median_bout_duration_s", "float64", True),
        ("mean_bout_duration_s", "float64", True),
        ("median_bout_path_length_mm", "float64", True),
        ("mean_bout_path_length_mm", "float64", True),
        ("median_abs_bout_net_heading_change_deg", "float64", True),
        ("mean_abs_bout_net_heading_change_deg", "float64", True),
        ("median_inter_bout_interval_s", "float64", True),
        ("mean_inter_bout_interval_s", "float64", True),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("collection_manifest_path", "string", True),
    )


@pytest.mark.parametrize(
    "table_name",
    (
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        RECORDING_SUMMARY_TABLE,
        BASELINE_BEHAVIOR_SUMMARY_TABLE,
    ),
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


def test_baseline_summary_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = BASELINE_BEHAVIOR_SUMMARY_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_baseline_summary_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert (
        schema.remove_metadata()
        == exact_arrow_schema(
            table_name,
            metadata={},
        ).remove_metadata()
    )
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


def test_baseline_summary_exact_writer_rejects_unexpected_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    row = _valid_baseline_summary_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=BASELINE_BEHAVIOR_SUMMARY_TABLE,
            rows_by_source=(("source", [row]),),
        )

    required = {
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE].fields
        if not field.nullable
    }
    for field_name in sorted(required):
        row = _valid_baseline_summary_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=BASELINE_BEHAVIOR_SUMMARY_TABLE,
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


def test_baseline_summary_zero_rows_publish_no_parts_but_retain_exact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={BASELINE_BEHAVIOR_SUMMARY_TABLE: []},
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
        export_run_id="empty-baseline-summary",
        tables=(BASELINE_BEHAVIOR_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_BEHAVIOR_SUMMARY_TABLE: 0}
    assert manifest["part_files_by_table"] == {BASELINE_BEHAVIOR_SUMMARY_TABLE: []}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        BASELINE_BEHAVIOR_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-baseline-summary")["status"] == "valid"


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


def test_real_baseline_summary_export_uses_exact_schema_without_promoting_source_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "baseline_recording_analysis.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=3)
    analysis = root.create_group("analysis")
    calibration = analysis.create_group("calibration")
    calibration.attrs.update(
        {
            "experimental_area_shape": "circle",
            "experimental_area_center_x_px": 10.0,
            "experimental_area_center_y_px": 10.0,
            "experimental_area_radius_px": 10.0,
        }
    )
    chaser_parent = analysis.create_group("chaser_distance_runs")
    chaser = chaser_parent.create_group("run-1")
    chaser.attrs.update(
        {
            "schema_id": "palette.chaser.distance.v1",
            "schema_version": 1,
            "method": "fixture",
            "method_version": "1",
            "coordinate_frame": "projector_canvas_px",
            "coordinate_origin": "top_left",
            "total_frames": 20,
            "pixels_per_mm_projector": 1.0,
            "source_refs": {},
        }
    )
    positions = chaser.create_group("positions")
    positions.create_array(
        "fish_centroid_arena_xy",
        data=np.column_stack([np.linspace(10.0, 19.0, 20), np.full(20, 10.0)]),
    )
    positions.create_array("fish_valid", data=np.ones(20, dtype=bool))

    components = chaser.create_group("epoch_behavior_summary")
    components.attrs.update({"latest": "component-1", "latest_complete": "component-1"})
    component = components.create_group("component-1")
    component.attrs.update(
        {
            "status": "complete",
            "schema_id": "palette.chaser.epoch_behavior_summary.v1",
            "source_refs": {
                "source_track_kinematics_run": "track-1",
                "source_track_kinematics_scope": "offline",
                "source_track_kinematics_track_id": 0,
            },
            "parameters": {"speed_level": "filtered", "wall_band_mm": 2.0},
        }
    )
    summary = np.asarray(
        [
            (
                b"pre_event",
                0,
                0,
                9,
                0.0,
                1.0,
                1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                999.0,
            )
        ],
        dtype=[
            ("window_label", "S32"),
            ("window_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("start_time_s", "f8"),
            ("end_time_s", "f8"),
            ("duration_s", "f8"),
            ("median_bout_duration_s", "f8"),
            ("mean_bout_duration_s", "f8"),
            ("median_bout_path_length_mm", "f8"),
            ("mean_bout_path_length_mm", "f8"),
            ("median_abs_bout_net_heading_change_deg", "f8"),
            ("mean_abs_bout_net_heading_change_deg", "f8"),
            ("median_inter_bout_interval_s", "f8"),
            ("mean_inter_bout_interval_s", "f8"),
            ("future_source_metric", "f8"),
        ],
    )
    write_columnar_dataset(component, "per_epoch_fish", summary, shard_rows=None)

    def latest_run(
        opened_root: Any,
        parent_path: str,
        requested: str | None = None,
    ) -> tuple[Any | None, str | None, str | None]:
        if parent_path == "analysis/chaser_distance_runs":
            assert requested is None
            return opened_root[parent_path]["run-1"], "run-1", None
        return None, None, "fixture has no requested run"

    track = SimpleNamespace(
        run_name="track-1",
        scope="offline",
        run_path="analysis/track_kinematics_runs/track-1",
        track_path="analysis/track_kinematics_runs/track-1/tracks/id_0",
        run_attrs={"fps": 10.0},
        frame_indices=np.arange(20, dtype=np.int64),
        time_seconds=np.arange(20, dtype=np.float64) / 10.0,
        speed_mm_by_level={"filtered": np.arange(20, dtype=np.float64)},
        frame_path_distance_mm_by_level={"filtered": np.ones(20, dtype=np.float64)},
        smoothed_heading_degrees=np.linspace(-45.0, 45.0, 20),
        heading_degrees=np.linspace(-45.0, 45.0, 20),
        sample_valid=np.ones(20, dtype=bool),
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics._latest_run",
        latest_run,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.load_track_kinematics_track",
        lambda *_args, **_kwargs: track,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )

    export_root = tmp_path / "exports"
    manifest = export_sources(
        [source],
        output_root=export_root,
        export_run_id="baseline-summary-arrow",
        tables=(BASELINE_BEHAVIOR_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_BEHAVIOR_SUMMARY_TABLE: 1}
    part = (
        export_root
        / manifest["part_files_by_table"][BASELINE_BEHAVIOR_SUMMARY_TABLE][0]
    )
    parquet_file = pq.ParquetFile(part)
    table = parquet_file.read()
    validate_arrow_schema(BASELINE_BEHAVIOR_SUMMARY_TABLE, parquet_file.schema_arrow)
    assert parquet_file.schema_arrow.names == [
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE].fields
    ]
    row = table.to_pylist()[0]
    assert row["recording_id"] == "baseline_recording"
    assert row["fps"] is None
    assert row["median_bout_duration_s"] == 1.0
    assert "future_source_metric" not in row
    assert (
        validate_export_run(export_root, "baseline-summary-arrow")["status"] == "valid"
    )


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


@pytest.mark.parametrize(
    "mutation",
    (
        "reordered",
        "wrong_type",
        "wrong_nullability",
        "unexpected",
        "missing",
        "metadata",
    ),
)
def test_baseline_summary_manifest_reader_rejects_rehashed_physical_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    table_name = BASELINE_BEHAVIOR_SUMMARY_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_baseline_summary_row()]},
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
        export_run_id=f"baseline-summary-{mutation}",
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
        index = original.schema.get_field_index("total_frame_count")
        arrays[index] = pa.array([1.0], type=pa.float64())
        fields[index] = pa.field("total_frame_count", pa.float64(), nullable=False)
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

    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata)),
        part,
    )
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
        validate_export_run(root, f"baseline-summary-{mutation}")
