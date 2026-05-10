from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.utils.export_cross_recording_analytics import export_sources
from fisheye.utils.export_cross_recording_analytics import main as export_main
from fisheye.utils.virtual_collection_manifest import with_manifest_sha256


def _array(group, name: str, values) -> None:
    group.create_array(name, data=np.asarray(values), overwrite=True)


def _make_source_zarr(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w")
    analysis = root.create_group("analysis")

    stim_parent = analysis.create_group("stimulus_runs")
    stim_parent.attrs["latest"] = "stimulus_test"
    stim = stim_parent.create_group("stimulus_test")
    steps = stim.create_group("steps")
    step0 = steps.create_group("step_0")
    step0.attrs.update(
        {
            "step_index": 0,
            "step_name": "Moving Grating",
            "stimulus_mode": "MOVING_GRATING",
            "stimulus_mode_id": 3,
            "start_frame": 10,
            "end_frame": 110,
            "duration_s": 1.6667,
            "stimulus_params": {"direction_degrees": 0.0},
        }
    )
    moving = step0.create_group("moving_grating")
    moving.attrs["direction_degrees"] = 0.0
    moving.attrs["direction_mapping_status"] = "validated"

    step1 = steps.create_group("step_1")
    step1.attrs.update(
        {
            "step_index": 1,
            "step_name": "Concentric",
            "stimulus_mode": "CONCENTRIC_GRATING",
            "stimulus_mode_id": 6,
            "start_frame": 110,
            "end_frame": 210,
            "duration_s": 1.6667,
            "stimulus_params": {"is_expanding": False},
        }
    )
    concentric = step1.create_group("concentric_grating")
    concentric.attrs["stimulus_radial_polarity_authored"] = "contracting"

    resp_parent = analysis.create_group("stimulus_response_runs")
    resp_parent.attrs["latest"] = "stimulus_response_test"
    resp = resp_parent.create_group("stimulus_response_test")
    resp.attrs.update(
        {
            "source_stimulus_run": "stimulus_test",
            "source_track_kinematics_run": "tk_test",
            "source_track_kinematics_type": "offline",
            "source_bout_run": "bouts_test",
            "n_fish": 1,
            "n_steps": 2,
        }
    )
    global_group = resp.create_group("global")
    _array(global_group, "fish_id", [0])
    _array(global_group, "total_distance_mm", [25.0])
    _array(global_group, "mean_speed_mm_s", [3.5])
    _array(global_group, "total_active_s", [4.0])
    _array(global_group, "fraction_moving", [0.25])

    resp_steps = resp.create_group("steps")
    resp_step0 = resp_steps.create_group("step_0")
    resp_step0.attrs.update(dict(step0.attrs))
    pf0 = resp_step0.create_group("per_fish")
    _array(pf0, "fish_id", [0])
    _array(pf0, "total_distance_mm", [12.5])
    _array(pf0, "mean_speed_mm_s", [5.0])
    _array(pf0, "num_bouts", [2])
    grating = resp_step0.create_group("grating")
    omr = grating.create_group("omr")
    omr.attrs["method_version"] = "omr.v1"
    omr_pf = omr.create_group("per_fish")
    _array(omr_pf, "fish_id", [0])
    _array(omr_pf, "omr_path_index", [0.75])
    _array(omr_pf, "first_aligned_bout_latency_s", [np.nan])

    resp_step1 = resp_steps.create_group("step_1")
    resp_step1.attrs.update(dict(step1.attrs))
    pf1 = resp_step1.create_group("per_fish")
    _array(pf1, "fish_id", [0])
    _array(pf1, "total_distance_mm", [10.0])
    _array(pf1, "mean_speed_mm_s", [4.0])
    _array(pf1, "num_bouts", [1])
    conc = resp_step1.create_group("concentric_grating")
    radial = conc.create_group("radial_omr")
    radial.attrs["method_version"] = "radial.v1"
    radial_pf = radial.create_group("per_fish")
    _array(radial_pf, "fish_id", [0])
    _array(radial_pf, "omr_path_index", [0.5])
    _array(radial_pf, "radial_path_index", [-0.5])
    _array(radial_pf, "first_aligned_bout_latency_s", [0.2])

    swim_parent = analysis.create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "bouts_test"
    swim = swim_parent.create_group("bouts_test")
    swim.attrs.update(
        {
            "default_level": "speed_exponential",
            "track_id": 0,
            "source_track_kinematics_run": "tk_test",
            "source_track_kinematics_type": "offline",
        }
    )
    level = swim.create_group("speed_exponential")
    level.attrs.update(
        {
            "n_bouts": 2,
            "mean_bout_duration_s": 0.12,
            "total_path_length_mm": 8.0,
            "detection_method": "peak_event",
            "detection_signal_transform_type": "causal_exponential",
            "movement_metric_source_level": "filtered",
            "peak_prominence_mm_s": 4.0,
        }
    )
    bouts = level.create_group("bouts")
    _array(bouts, "bout_id", [0, 1])
    _array(bouts, "start_frame", [20, 140])
    _array(bouts, "end_frame", [30, 150])
    _array(bouts, "start_time_s", [0.3333, 2.3333])
    _array(bouts, "end_time_s", [0.5, 2.5])
    _array(bouts, "duration_s", [0.1667, 0.1667])
    _array(bouts, "path_length_mm", [3.0, 5.0])
    _array(bouts, "net_displacement_mm", [2.0, 4.0])
    _array(bouts, "mean_speed_mm_s", [18.0, 30.0])
    _array(bouts, "peak_physical_speed_mm_s", [25.0, 45.0])

    bout_kin_parent = analysis.create_group("bout_kinematics_runs")
    bout_kin_parent.attrs["latest"] = "bout_kinematics_test"
    bout_kin = bout_kin_parent.create_group("bout_kinematics_test")
    bout_kin.attrs.update(
        {
            "schema_version": 7,
            "method": "bout_kinematics",
            "method_version": "bout_kinematics.v7",
            "source_track_id": 0,
            "source_track_kinematics_run": "tk_test",
            "source_swim_bout_run": "bouts_test",
            "source_swim_bout_speed_level": "speed_exponential",
            "default_heading_level": "heading_smoothed",
        }
    )

    movement_metrics = bout_kin.create_group("movement").create_group("per_bout_metrics")
    _array(movement_metrics, "bout_id", [0, 1])
    _array(movement_metrics, "source_start_frame", [20, 140])
    _array(movement_metrics, "source_end_frame", [30, 150])
    _array(movement_metrics, "physical_active_duration_s", [0.12, 0.14])
    _array(movement_metrics, "physical_active_path_length_mm", [2.5, 4.5])
    _array(movement_metrics, "physical_active_valid", [True, True])

    for level_name, deltas in (
        ("heading_smoothed", [12.5, -30.0]),
        ("heading_raw", [14.0, -28.0]),
    ):
        level_group = bout_kin.create_group(level_name)
        level_group.attrs["is_default_heading_level"] = level_name == "heading_smoothed"
        metrics = level_group.create_group("per_bout_metrics")
        _array(metrics, "bout_id", [0, 1])
        _array(metrics, "source_start_frame", [20, 140])
        _array(metrics, "source_end_frame", [30, 150])
        _array(metrics, "pre_heading_mean_deg", [5.0, 40.0])
        _array(metrics, "post_heading_mean_deg", [17.5, 10.0])
        _array(metrics, "net_delta_heading_deg", deltas)
        _array(metrics, "abs_net_delta_heading_deg", np.abs(deltas))
        _array(metrics, "within_heading_path_deg", [18.0, 36.0])
        _array(metrics, "within_heading_peak_to_peak_deg", [15.0, 32.0])
        _array(metrics, "within_angular_speed_mean_deg_s", [90.0, 120.0])
        _array(metrics, "within_angular_speed_max_deg_s", [250.0, 350.0])
        _array(metrics, "within_window_valid", [True, True])

    return path


def _convert_bout_kinematics_fixture_to_compact_v2(path: Path) -> None:
    root = zarr.open_group(str(path), mode="a")
    bout_kin = root["analysis"]["bout_kinematics_runs"]["bout_kinematics_test"]
    for name in ("movement", "heading_smoothed", "heading_raw"):
        del bout_kin[name]

    bout_kin.attrs["layout"] = "compact_tabular_v2"
    bout_kin.attrs["analysis_levels"] = ["movement", "heading_smoothed", "heading_raw"]
    bout_kin.attrs["heading_levels"] = ["heading_smoothed", "heading_raw"]

    level_index = np.asarray(
        [
            (0, b"movement", b"movement", -1, b"", False, 2),
            (1, b"heading_smoothed", b"heading", 0, b"heading_smoothed", True, 2),
            (2, b"heading_raw", b"heading", 1, b"heading_raw", False, 2),
        ],
        dtype=[
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S64"),
            ("measurement_family_bytes", "S64"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S64"),
            ("is_default_heading_level", "?"),
            ("row_count", "i8"),
        ],
    )
    write_columnar_dataset(
        bout_kin,
        "level_index",
        level_index,
        attrs={"schema_version": 7, "layout": "compact_tabular_v2"},
    )

    movement = np.asarray(
        [
            (0, b"movement", -1, b"", 0, 20, 30, 0.12, 2.5, True),
            (0, b"movement", -1, b"", 1, 140, 150, 0.14, 4.5, True),
        ],
        dtype=[
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S64"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S64"),
            ("bout_id", "i4"),
            ("source_start_frame", "i8"),
            ("source_end_frame", "i8"),
            ("physical_active_duration_s", "f8"),
            ("physical_active_path_length_mm", "f8"),
            ("physical_active_valid", "?"),
        ],
    )
    write_columnar_dataset(
        bout_kin,
        "movement_metrics",
        movement,
        attrs={"schema_version": 7, "layout": "compact_tabular_v2", "analysis_level": "movement"},
    )

    heading_rows = []
    for level_id, level_name, deltas in (
        (0, b"heading_smoothed", [12.5, -30.0]),
        (1, b"heading_raw", [14.0, -28.0]),
    ):
        for bout_id, start, end, delta in zip((0, 1), (20, 140), (30, 150), deltas):
            heading_rows.append(
                (
                    level_id + 1,
                    level_name,
                    level_id,
                    level_name,
                    bout_id,
                    start,
                    end,
                    5.0 if bout_id == 0 else 40.0,
                    17.5 if bout_id == 0 else 10.0,
                    delta,
                    abs(delta),
                    18.0 if bout_id == 0 else 36.0,
                    15.0 if bout_id == 0 else 32.0,
                    90.0 if bout_id == 0 else 120.0,
                    250.0 if bout_id == 0 else 350.0,
                    True,
                )
            )
    heading = np.asarray(
        heading_rows,
        dtype=[
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S64"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S64"),
            ("bout_id", "i4"),
            ("source_start_frame", "i8"),
            ("source_end_frame", "i8"),
            ("pre_heading_mean_deg", "f8"),
            ("post_heading_mean_deg", "f8"),
            ("net_delta_heading_deg", "f8"),
            ("abs_net_delta_heading_deg", "f8"),
            ("within_heading_path_deg", "f8"),
            ("within_heading_peak_to_peak_deg", "f8"),
            ("within_angular_speed_mean_deg_s", "f8"),
            ("within_angular_speed_max_deg_s", "f8"),
            ("within_window_valid", "?"),
        ],
    )
    write_columnar_dataset(
        bout_kin,
        "heading_metrics",
        heading,
        attrs={
            "schema_version": 7,
            "layout": "compact_tabular_v2",
            "analysis_level": "heading",
            "heading_levels": ["heading_smoothed", "heading_raw"],
            "default_heading_level": "heading_smoothed",
        },
    )


def _read_dataset(output_root: Path, table: str, export_run_id: str):
    table_dir = output_root / "v1" / table / f"export_run_id={export_run_id}"
    files = sorted(table_dir.glob("*.parquet"))
    assert files, f"no parquet files for {table}"
    return pq.read_table(files).to_pylist()


def _write_collection_manifest(path: Path, source: Path) -> dict:
    payload = with_manifest_sha256(
        {
            "schema_id": "palette.virtual_collection_manifest",
            "schema_version": 1,
            "collection_id": "collection_test",
            "collection_name": "Collection Test",
            "records": [
                {
                    "recording_id": "recording_a",
                    "locator_at_selection": {"uri": str(source.resolve())},
                    "status": {"included": True},
                }
            ],
        }
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def test_export_cross_recording_analytics_writes_first_tables(tmp_path: Path) -> None:
    source = _make_source_zarr(tmp_path / "recording_a_analysis.zarr")
    output = tmp_path / "exports" / "palette_analytics"
    collection_manifest = _write_collection_manifest(tmp_path / "collection.manifest.json", source)

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="test_export",
        jobs=1,
        collection_manifest_path=tmp_path / "collection.manifest.json",
    )

    assert manifest["row_counts_by_table"]["recording_summary"] == 1
    assert manifest["row_counts_by_table"]["stimulus_steps"] == 2
    assert manifest["row_counts_by_table"]["stimulus_step_summary"] == 2
    assert manifest["row_counts_by_table"]["stimulus_response_per_fish_step"] == 2
    assert manifest["row_counts_by_table"]["swim_bout_metrics"] == 2
    assert manifest["row_counts_by_table"]["bout_kinematics_metrics"] == 6

    manifest_path = output / "v1" / "manifests" / "export_run_id=test_export.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["source_recording_count"] == 1
    assert payload["row_counts_by_table"]["swim_bout_metrics"] == 2
    assert payload["collection_manifest"]["collection_id"] == "collection_test"
    assert payload["collection_manifest"]["manifest_sha256"] == collection_manifest["manifest_sha256"]

    step_rows = _read_dataset(output, "stimulus_steps", "test_export")
    protocol_hash = step_rows[0]["protocol_signature_hash"]
    assert step_rows[0]["collection_id"] == "collection_test"
    assert step_rows[0]["collection_manifest_sha256"] == collection_manifest["manifest_sha256"]
    assert step_rows[0]["derived_protocol_hash"] == protocol_hash
    assert step_rows[1]["protocol_signature_hash"] == protocol_hash
    assert step_rows[0]["protocol_mode_sequence"] == "MOVING_GRATING -> CONCENTRIC_GRATING"

    response_rows = _read_dataset(output, "stimulus_response_per_fish_step", "test_export")
    moving = next(row for row in response_rows if row["stimulus_mode"] == "MOVING_GRATING")
    assert moving["protocol_signature_hash"] == protocol_hash
    assert isinstance(protocol_hash, str)
    assert len(protocol_hash) == 64
    assert moving["derived_protocol_hash"] == protocol_hash
    assert moving["protocol_signature_schema"] == "palette_protocol_signature_v1"
    assert moving["protocol_mode_sequence"] == "MOVING_GRATING -> CONCENTRIC_GRATING"
    assert moving["protocol_duration_sequence_s"] == "1.6667,1.6667"
    assert moving["protocol_step_count"] == 2
    assert moving["omr_family"] == "moving_grating_omr"
    assert moving["omr_path_index"] == 0.75
    assert moving["first_aligned_bout_latency_s"] is None

    radial = next(row for row in response_rows if row["stimulus_mode"] == "CONCENTRIC_GRATING")
    assert radial["omr_family"] == "concentric_radial_omr"
    assert radial["first_aligned_bout_latency_s"] == 0.2
    assert radial["radial_path_index"] == -0.5

    bout_rows = _read_dataset(output, "swim_bout_metrics", "test_export")
    assert bout_rows[0]["protocol_signature_hash"] == protocol_hash
    assert bout_rows[0]["derived_protocol_hash"] == protocol_hash
    assert bout_rows[0]["step_index"] == 0
    assert bout_rows[1]["step_index"] == 1
    assert bout_rows[0]["speed_level"] == "speed_exponential"

    bout_kin_rows = _read_dataset(output, "bout_kinematics_metrics", "test_export")
    assert len(bout_kin_rows) == 6
    heading_rows = [row for row in bout_kin_rows if row["measurement_level"] == "heading_smoothed"]
    assert len(heading_rows) == 2
    assert heading_rows[0]["measurement_family"] == "heading"
    assert heading_rows[0]["is_default_heading_level"] is True
    assert heading_rows[0]["source_swim_bout_run"] == "bouts_test"
    assert heading_rows[0]["source_swim_bout_speed_level"] == "speed_exponential"
    assert heading_rows[0]["protocol_signature_hash"] == protocol_hash
    assert heading_rows[0]["derived_protocol_hash"] == protocol_hash
    assert heading_rows[0]["step_index"] == 0
    assert heading_rows[1]["step_index"] == 1
    assert heading_rows[0]["net_delta_heading_deg"] == 12.5
    assert heading_rows[1]["abs_net_delta_heading_deg"] == 30.0
    movement_rows = [row for row in bout_kin_rows if row["measurement_level"] == "movement"]
    assert len(movement_rows) == 2
    assert movement_rows[0]["measurement_family"] == "movement"
    assert movement_rows[0]["physical_active_duration_s"] == 0.12


def test_export_cross_recording_analytics_reads_compact_bout_kinematics(tmp_path: Path) -> None:
    source = _make_source_zarr(tmp_path / "recording_compact_analysis.zarr")
    _convert_bout_kinematics_fixture_to_compact_v2(source)
    output = tmp_path / "exports" / "palette_analytics"

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="compact_export",
        tables=("bout_kinematics_metrics",),
        jobs=1,
    )

    assert manifest["row_counts_by_table"]["bout_kinematics_metrics"] == 6
    rows = _read_dataset(output, "bout_kinematics_metrics", "compact_export")
    assert len(rows) == 6
    assert {row["measurement_level"] for row in rows} == {"movement", "heading_smoothed", "heading_raw"}
    heading_rows = [row for row in rows if row["measurement_level"] == "heading_smoothed"]
    assert len(heading_rows) == 2
    assert heading_rows[0]["measurement_family"] == "heading"
    assert heading_rows[0]["is_default_heading_level"] is True
    assert heading_rows[0]["net_delta_heading_deg"] == 12.5
    assert "analysis_level_bytes" not in heading_rows[0]
    movement_rows = [row for row in rows if row["measurement_level"] == "movement"]
    assert movement_rows[0]["physical_active_path_length_mm"] == 2.5


def test_export_cross_recording_analytics_can_limit_tables(tmp_path: Path) -> None:
    source = _make_source_zarr(tmp_path / "recording_b_analysis.zarr")
    output = tmp_path / "exports" / "palette_analytics"

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="summary_only",
        tables=("recording_summary",),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {"recording_summary": 1}
    assert (output / "v1" / "recording_summary" / "export_run_id=summary_only").is_dir()
    assert not (output / "v1" / "swim_bout_metrics").exists()
    rows = _read_dataset(output, "recording_summary", "summary_only")
    assert rows[0]["protocol_signature_schema"] == "palette_protocol_signature_v1"
    assert rows[0]["derived_protocol_hash"] == rows[0]["protocol_signature_hash"]
    assert rows[0]["protocol_step_count"] == 2


def test_export_cross_recording_analytics_can_index_registry(tmp_path: Path, capsys) -> None:
    source = _make_source_zarr(tmp_path / "recording_c_analysis.zarr")
    output = tmp_path / "exports" / "palette_analytics"
    registry_path = tmp_path / "registry.sqlite"
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path, source)

    export_main(
        [
            "--collection-manifest",
            str(collection_path),
            "--output-root",
            str(output),
            "--tables",
            "recording_summary",
            "--jobs",
            "1",
            "--export-run-id",
            "indexed_export",
            "--registry",
            str(registry_path),
            "--index-registry",
        ]
    )

    stdout = capsys.readouterr().out
    assert f"indexed_registry\t{registry_path.resolve()}\tindexed_export" in stdout

    import sqlite3

    conn = sqlite3.connect(registry_path)
    try:
        export_row = conn.execute(
            """
            SELECT collection_id, collection_manifest_sha256, source_recording_count, table_count
            FROM analytics_export_overview
            WHERE export_run_id = 'indexed_export';
            """
        ).fetchone()
        assert export_row == ("collection_test", collection["manifest_sha256"], 1, 1)
        table_row = conn.execute(
            """
            SELECT table_name, row_count, part_count
            FROM analytics_export_tables
            WHERE export_run_id = 'indexed_export';
            """
        ).fetchone()
        assert table_row == ("recording_summary", 1, 1)
    finally:
        conn.close()
