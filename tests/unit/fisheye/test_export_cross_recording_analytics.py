from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
)
from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.chaser_egocentric_bearing import (
    build_chaser_egocentric_bearing_result,
    write_chaser_egocentric_bearing_component,
)
from fisheye.analysis.cra_primary_endpoint import (
    DEFAULT_COMPONENT_NAME as DEFAULT_CRA_COMPONENT_NAME,
    build_cra_primary_endpoint_result,
    write_cra_primary_endpoint_component,
)
from fisheye.analysis.cra_near_field import (
    DEFAULT_COMPONENT_NAME as DEFAULT_CRA_NEAR_FIELD_COMPONENT_NAME,
    build_cra_near_field_result,
    write_cra_near_field_component,
)
from fisheye.analysis.goodcopbadcop_epoch_behavior_summary import (
    build_goodcopbadcop_epoch_behavior_summary_result,
    write_goodcopbadcop_epoch_behavior_summary_component,
)
from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.analysis.stimulus_response import (
    ConcentricStepData,
    GratingStepData,
    ProtocolStep,
    STIMULUS_RESPONSE_LAYOUT_COMPACT_V2,
    write_stimulus_response_run,
)
from fisheye.analysis.stimulus_response_concentric_omr import ConcentricRadialOMRStepData
from fisheye.analysis.stimulus_response_omr import OMRStepData
from fisheye.utils.export_cross_recording_analytics import (
    _chaser_behaviors_for_run,
    export_sources,
)
from fisheye.utils.export_cross_recording_analytics import main as export_main
from fisheye.utils.virtual_collection_manifest import with_manifest_sha256
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
    _make_chaser_result,
)
from tests.unit.fisheye.test_chaser_egocentric_bearing import _add_track_kinematics_run
from tests.unit.fisheye.test_cra_near_field import _add_circle_geometry


def _array(group, name: str, values) -> None:
    group.create_array(name, data=np.asarray(values), overwrite=True)


def test_exporter_replaces_unknown_chaser_roles_from_cra_objects() -> None:
    run = zarr.group()
    chasers = run.create_group("chasers")
    _array(chasers, "chaser_index", [0, 1])
    _array(chasers, "behavior_class_id", [0, 0])
    _array(
        chasers,
        "behavior_class_label_bytes",
        np.asarray([b"unknown", b"unknown"], dtype="S16"),
    )
    parent = run.create_group("cra_primary_endpoint")
    parent.attrs["latest_complete"] = "roles"
    component = parent.create_group("roles")
    component.attrs["status"] = "computed"
    objects = component.create_group("objects")
    _array(objects, "object_index", [0, 1])
    _array(objects, "behavior_class_id", [1, 3])
    _array(
        objects,
        "behavior_class_label_bytes",
        np.asarray([b"aggressive", b"inert"], dtype="S16"),
    )

    assert _chaser_behaviors_for_run(run, [0, 1]) == [
        (1, "aggressive"),
        (3, "inert"),
    ]


def _add_goodcopbadcop_cra_protocol_metadata(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    stimulus = root["analysis/stimulus_runs/stimulus_1"]
    stimulus.attrs["protocol_json"] = json.dumps(
        {
            "steps": [
                {
                    "parameters": {
                        "position_transition_duration_s": 0.1,
                        "pre_period_duration_s": 0.3,
                        "training_period_duration_s": 0.3,
                        "post_period_duration_s": 0.3,
                        "pixels_per_mm": 2.0,
                        "chasers": [
                            {
                                "enable_chase": True,
                                "behavior_mode": 0,
                                "color_r": 1.0,
                                "color_g": 0.0,
                                "color_b": 0.0,
                                "color_a": 1.0,
                                "start_position_preset": "top_left",
                                "end_position_preset": "bottom_right",
                            },
                            {
                                "enable_chase": False,
                                "behavior_mode": 1,
                                "color_r": 0.0,
                                "color_g": 0.0,
                                "color_b": 1.0,
                                "color_a": 1.0,
                                "start_position_preset": "top_right",
                                "end_position_preset": "bottom_left",
                            },
                        ],
                    }
                }
            ]
        }
    )
    coords = stimulus.require_group("stimulus_coordinates")
    if "arena_1" in coords:
        del coords["arena_1"]
    arena = coords.create_group("arena_1")
    arena.attrs.update(
        {
            "texture_width_px": 20.0,
            "texture_height_px": 20.0,
            "texture_origin": "top_left",
        }
    )


def _add_goodcopbadcop_swim_bout_run(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    track = root["analysis/track_kinematics_runs/offline/tk_1/tracks/id_0"]
    track.create_array(
        "speed_filtered_mm",
        data=np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32),
        chunks=(8,),
        overwrite=True,
    )
    tk_run = root["analysis/track_kinematics_runs/offline/tk_1"]
    tk_run.attrs["fps"] = 10.0
    tk_run.attrs["pixel_to_mm"] = 0.02

    parent = root["analysis"].require_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_1"
    run = parent.create_group("bouts_1")
    run.attrs.update(
        {
            "default_level": "filtered",
            "source_track_kinematics_run": "tk_1",
            "track_id": 0,
            "detection_method": "threshold",
        }
    )
    level = run.create_group("speed_filtered")
    level.attrs["n_bouts"] = 4
    bouts = np.zeros(
        4,
        dtype=[
            ("bout_id", np.int32),
            ("peak_time_s", np.float64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("duration_s", np.float64),
            ("path_length_mm", np.float64),
        ],
    )
    bouts["bout_id"] = [0, 1, 2, 3]
    bouts["peak_time_s"] = [0.10, 0.20, 0.45, 0.70]
    bouts["start_time_s"] = [0.08, 0.18, 0.42, 0.68]
    bouts["end_time_s"] = [0.12, 0.24, 0.50, 0.76]
    bouts["start_frame"] = [0, 1, 4, 7]
    bouts["end_frame"] = [1, 2, 5, 8]
    bouts["duration_s"] = [0.04, 0.06, 0.08, 0.08]
    bouts["path_length_mm"] = [0.2, 0.3, 0.4, 0.5]
    write_columnar_dataset(level, "bouts", bouts, {"n_bouts": 4})

    intervals = np.zeros(
        2,
        dtype=[
            ("interval_id", np.int32),
            ("valid", bool),
            ("prev_end_time_s", np.float64),
            ("next_start_time_s", np.float64),
            ("interval_s", np.float64),
        ],
    )
    intervals["interval_id"] = [0, 1]
    intervals["valid"] = [True, True]
    intervals["prev_end_time_s"] = [0.12, 0.50]
    intervals["next_start_time_s"] = [0.18, 0.68]
    intervals["interval_s"] = [0.06, 0.18]
    write_columnar_dataset(level, "inter_bout_intervals", intervals, {"n_intervals": 2})


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


def _replace_stimulus_response_fixture_with_compact_v2(path: Path) -> None:
    root = zarr.open_group(str(path), mode="a")
    steps = [
        ProtocolStep(
            0,
            "Moving Grating",
            "MOVING_GRATING",
            3,
            10,
            110,
            1.6667,
            {"direction_degrees": 0.0},
        ),
        ProtocolStep(
            1,
            "Concentric",
            "CONCENTRIC_GRATING",
            6,
            110,
            210,
            1.6667,
            {"is_expanding": False},
        ),
    ]
    global_metrics = {
        "fish_id": np.asarray([0], dtype=np.int32),
        "total_distance_mm": np.asarray([25.0], dtype=np.float32),
        "mean_speed_mm_s": np.asarray([3.5], dtype=np.float32),
        "total_active_s": np.asarray([4.0], dtype=np.float32),
        "fraction_moving": np.asarray([0.25], dtype=np.float32),
    }
    step_metrics = [
        {
            "fish_id": np.asarray([0], dtype=np.int32),
            "total_distance_mm": np.asarray([12.5], dtype=np.float32),
            "mean_speed_mm_s": np.asarray([5.0], dtype=np.float32),
            "num_bouts": np.asarray([2], dtype=np.int32),
        },
        {
            "fish_id": np.asarray([0], dtype=np.int32),
            "total_distance_mm": np.asarray([10.0], dtype=np.float32),
            "mean_speed_mm_s": np.asarray([4.0], dtype=np.float32),
            "num_bouts": np.asarray([1], dtype=np.int32),
        },
    ]
    grating = GratingStepData(
        per_frame={},
        per_fish={},
        time_series={},
        omr=OMRStepData(
            per_fish={
                "fish_id": np.asarray([0], dtype=np.int32),
                "omr_path_index": np.asarray([0.75], dtype=np.float32),
                "first_aligned_bout_latency_s": np.asarray([np.nan], dtype=np.float32),
            },
            per_bout={},
            windows={},
            early_windows={},
            attrs={"method_version": "omr.v1"},
        ),
    )
    concentric = ConcentricStepData(
        per_frame={},
        per_fish={},
        time_series={},
        radial_omr=ConcentricRadialOMRStepData(
            per_frame={},
            per_fish={
                "fish_id": np.asarray([0], dtype=np.int32),
                "omr_path_index": np.asarray([0.5], dtype=np.float32),
                "radial_path_index": np.asarray([-0.5], dtype=np.float32),
                "first_aligned_bout_latency_s": np.asarray([0.2], dtype=np.float32),
            },
            per_bout={},
            windows={},
            early_windows={},
            attrs={"method_version": "radial.v1"},
        ),
    )
    write_stimulus_response_run(
        root,
        global_metrics=global_metrics,
        steps=steps,
        step_metrics=step_metrics,
        step_grating_data={0: grating},
        step_concentric_data={1: concentric},
        source_kinematics_run="tk_test",
        source_kinematics_type="offline",
        source_stimulus_run="stimulus_test",
        source_bout_run="bouts_test",
        parameters={"layout": STIMULUS_RESPONSE_LAYOUT_COMPACT_V2},
        run_name="stimulus_response_test",
        overwrite=True,
        layout=STIMULUS_RESPONSE_LAYOUT_COMPACT_V2,
    )


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
    np.testing.assert_allclose(moving["omr_path_index"], 0.75)
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


def test_export_cross_recording_analytics_reads_goodcopbadcop_tables(tmp_path: Path) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    _add_goodcopbadcop_cra_protocol_metadata(source)
    write_chaser_distance_run(source, _make_chaser_result(source), overwrite=True)
    cra_result = build_cra_primary_endpoint_result(source, chaser_distance_run="chaser_distance_1")
    write_cra_primary_endpoint_component(source, cra_result, overwrite=True)
    _add_circle_geometry(source)
    near_field_result = build_cra_near_field_result(
        source,
        chaser_distance_run="chaser_distance_1",
        cra_primary_endpoint_component="object_relative_pre_post_v1",
        r_zone_mm=2.0,
        r_in_mm=2.0,
        r_out_mm=3.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0, 8.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=2.0,
    )
    write_cra_near_field_component(source, near_field_result, overwrite=True)
    _add_track_kinematics_run(source)
    _add_goodcopbadcop_swim_bout_run(source)
    epoch_behavior_result = build_goodcopbadcop_epoch_behavior_summary_result(
        source,
        chaser_distance_run="chaser_distance_1",
    )
    write_goodcopbadcop_epoch_behavior_summary_component(source, epoch_behavior_result, overwrite=True)
    egocentric_result = build_chaser_egocentric_bearing_result(
        source,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
        distance_bin_width_mm=2.0,
        bearing_bin_width_deg=90.0,
    )
    write_chaser_egocentric_bearing_component(source, egocentric_result, overwrite=True)
    output = tmp_path / "exports" / "palette_analytics"

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="goodcopbadcop_export",
        tables=(
            "position_occupancy_histogram_2d",
            "chaser_epoch_spatial_occupancy_zones",
            "chaser_epoch_distance_summary",
            "chaser_epoch_behavior_summary",
            "chaser_epoch_bout_events",
            "chaser_epoch_bout_histogram",
            "chaser_epoch_inter_bout_interval_histogram",
            "chaser_epoch_center_distance_histogram",
            "chaser_speed_distance_bins",
            "chaser_epoch_distance_histogram",
            "chaser_cra_primary_endpoint_summary",
            "chaser_cra_primary_endpoint_object_phase",
            "chaser_cra_quadrant_occupancy",
            "chaser_cra_near_field_summary",
            "chaser_cra_near_field_object_phase",
            "chaser_cra_near_field_radial_density",
            "chaser_cra_near_field_distance_cdf",
            "chaser_egocentric_epoch_summary",
            "chaser_egocentric_distance_bearing_histogram",
        ),
        jobs=1,
    )

    assert manifest["row_counts_by_table"]["position_occupancy_histogram_2d"] == 12
    assert manifest["row_counts_by_table"]["chaser_epoch_spatial_occupancy_zones"] == 12
    assert manifest["row_counts_by_table"]["chaser_epoch_distance_summary"] == 6
    assert manifest["row_counts_by_table"]["chaser_epoch_behavior_summary"] == 3
    assert manifest["row_counts_by_table"]["chaser_epoch_bout_events"] == 4
    assert manifest["row_counts_by_table"]["chaser_epoch_bout_histogram"] == 183
    assert manifest["row_counts_by_table"]["chaser_epoch_inter_bout_interval_histogram"] == 3
    assert manifest["row_counts_by_table"]["chaser_epoch_center_distance_histogram"] == 9
    assert manifest["row_counts_by_table"]["chaser_speed_distance_bins"] == 18
    assert manifest["row_counts_by_table"]["chaser_epoch_distance_histogram"] == 18
    assert manifest["row_counts_by_table"]["chaser_cra_primary_endpoint_summary"] == 1
    assert manifest["row_counts_by_table"]["chaser_cra_primary_endpoint_object_phase"] == 4
    assert manifest["row_counts_by_table"]["chaser_cra_quadrant_occupancy"] == 8
    assert manifest["row_counts_by_table"]["chaser_cra_near_field_summary"] == 1
    assert manifest["row_counts_by_table"]["chaser_cra_near_field_object_phase"] == 4
    assert manifest["row_counts_by_table"]["chaser_cra_near_field_radial_density"] == 12
    assert manifest["row_counts_by_table"]["chaser_cra_near_field_distance_cdf"] == 8
    assert manifest["row_counts_by_table"]["chaser_egocentric_epoch_summary"] == 6
    assert manifest["row_counts_by_table"]["chaser_egocentric_distance_bearing_histogram"] == 72
    assert manifest["schema_id"] == EXPORT_SCHEMA_ID
    assert manifest["schema_version"] == EXPORT_SCHEMA_VERSION
    assert "chaser.epoch.behavior_summary" in manifest["capabilities"]
    assert "position.epoch.occupancy_histogram_2d" in manifest["capabilities"]
    assert "chaser.cra.primary" in manifest["capabilities"]
    assert "chaser.egocentric" in manifest["capabilities"]
    assert set(manifest["table_contracts"]) == set(manifest["tables_requested"])

    first_part = Path(
        manifest["part_files_by_table"]["chaser_epoch_behavior_summary"][0]
    )
    schema_metadata = pq.ParquetFile(first_part).schema_arrow.metadata or {}
    assert schema_metadata[b"palette.export_schema_id"].decode() == EXPORT_SCHEMA_ID
    assert schema_metadata[b"palette.export_schema_version"].decode() == str(
        EXPORT_SCHEMA_VERSION
    )
    assert json.loads(schema_metadata[b"palette.table_contract"]) == TABLE_CONTRACTS[
        "chaser_epoch_behavior_summary"
    ].to_dict()

    position_rows = _read_dataset(
        output,
        "position_occupancy_histogram_2d",
        "goodcopbadcop_export",
    )
    pre_position_top_left = next(
        row
        for row in position_rows
        if row["window_label"] == "pre_event"
        and row["y_bin_index"] == 0
        and row["x_bin_index"] == 0
    )
    assert pre_position_top_left["hist_count"] == 1
    assert pre_position_top_left["coordinate_frame"] == "source_image_fraction"
    assert pre_position_top_left["coordinate_origin"] == "top_left"
    assert pre_position_top_left["x_bin_left_fraction"] == 0.0
    assert pre_position_top_left["x_bin_right_fraction"] == 0.5
    assert pre_position_top_left["y_bin_left_fraction"] == 0.0
    assert pre_position_top_left["y_bin_right_fraction"] == 0.5
    assert pre_position_top_left["image_width_px"] == 20.0
    assert pre_position_top_left["image_height_px"] == 20.0
    assert len(pre_position_top_left["normalized_grid_id"]) == 16

    spatial_rows = _read_dataset(
        output,
        "chaser_epoch_spatial_occupancy_zones",
        "goodcopbadcop_export",
    )
    pre_top_left = next(
        row
        for row in spatial_rows
        if row["window_label"] == "pre_event" and row["zone_id"] == "top_left"
    )
    assert pre_top_left["detection_occupancy_run"] == "occupancy_1"
    assert pre_top_left["detection_occupancy_path"] == "analysis/detection_occupancy_runs/occupancy_1"
    assert pre_top_left["source_detection_path"] == "refined_detect_runs/refined_1/instances"
    assert pre_top_left["zone_set_id"] == "image_quadrants_v1"
    assert pre_top_left["zone_set_source"] == "predefined_spec:quadrants.v1"
    assert pre_top_left["coordinate_frame"] == "source_image_px"
    assert pre_top_left["x_axis_direction"] == "right"
    assert pre_top_left["y_axis_direction"] == "down"
    assert pre_top_left["display_order"] == 0
    assert pre_top_left["x_min"] == 0.0
    assert pre_top_left["y_min"] == 0.0
    assert pre_top_left["x_max"] == 10.0
    assert pre_top_left["y_max"] == 10.0
    assert pre_top_left["frame_count"] == 1
    np.testing.assert_allclose(pre_top_left["time_s"], 0.1)
    np.testing.assert_allclose(pre_top_left["fraction_of_epoch"], 0.1)
    assert pre_top_left["detected_frame_count"] == 10
    assert len(pre_top_left["source_lineage_hash"]) == 64

    chaser_summary_rows = _read_dataset(
        output,
        "chaser_epoch_distance_summary",
        "goodcopbadcop_export",
    )
    post_chaser_1 = next(
        row
        for row in chaser_summary_rows
        if row["window_label"] == "post_event" and row["chaser_index"] == 1
    )
    assert post_chaser_1["chaser_distance_run"] == "chaser_distance_1"
    assert post_chaser_1["chaser_distance_schema_id"] == "palette.chaser_distance.v1"
    assert post_chaser_1["source_detection_path"] == "refined_detect_runs/refined_1/instances"
    assert post_chaser_1["source_stimulus_run"] == "stimulus_1"
    assert post_chaser_1["source_stimulus_epoch_run"] == "epochs_1"
    assert post_chaser_1["behavior_class_id"] == 3
    assert post_chaser_1["behavior_class"] == "inert"
    assert post_chaser_1["threshold_mm"] == 20.0
    assert post_chaser_1["valid_frame_count"] == 3
    assert post_chaser_1["p50_distance_mm"] == 6.0
    np.testing.assert_allclose(post_chaser_1["duration_s"], 0.3)

    epoch_behavior_rows = _read_dataset(
        output,
        "chaser_epoch_behavior_summary",
        "goodcopbadcop_export",
    )
    assert len(epoch_behavior_rows) == 3
    pre_behavior = next(row for row in epoch_behavior_rows if row["window_label"] == "pre_event")
    assert pre_behavior["epoch_behavior_component"] == "kinematics_bouts_v1"
    assert pre_behavior["epoch_behavior_path"].endswith("/epoch_behavior_summary/kinematics_bouts_v1")
    assert pre_behavior["source_swim_bout_run"] == "bouts_1"
    assert pre_behavior["source_track_kinematics_run"] == "tk_1"
    assert pre_behavior["source_speed_level"] == "filtered"
    assert pre_behavior["bout_count"] == 2
    assert "mean_bout_duration_s" in pre_behavior
    assert "mean_bout_path_length_mm" in pre_behavior
    assert "mean_bout_net_heading_change_deg" in pre_behavior
    assert "mean_abs_bout_net_heading_change_deg" in pre_behavior
    assert "wall_fraction" in pre_behavior
    assert pre_behavior["inter_bout_interval_count"] == 1
    np.testing.assert_allclose(pre_behavior["mean_inter_bout_interval_s"], 0.06)
    np.testing.assert_allclose(pre_behavior["mean_speed_mm_s"], 20.0)

    epoch_bout_rows = _read_dataset(
        output,
        "chaser_epoch_bout_events",
        "goodcopbadcop_export",
    )
    assert len(epoch_bout_rows) == 4
    pre_bout_rows = [row for row in epoch_bout_rows if row["window_label"] == "pre_event"]
    assert len(pre_bout_rows) == 2
    assert pre_bout_rows[0]["epoch_behavior_component"] == "kinematics_bouts_v1"
    assert pre_bout_rows[0]["source_swim_bout_run"] == "bouts_1"
    assert pre_bout_rows[0]["source_track_kinematics_run"] == "tk_1"
    assert pre_bout_rows[0]["bout_source_row"] == 0
    np.testing.assert_allclose(
        [row["bout_duration_s"] for row in pre_bout_rows],
        [0.04, 0.06],
    )
    np.testing.assert_allclose(
        [row["bout_path_length_mm"] for row in pre_bout_rows],
        [0.2, 0.3],
    )
    assert "bout_net_heading_change_deg" in pre_bout_rows[0]
    assert "abs_bout_net_heading_change_deg" in pre_bout_rows[0]

    bout_hist_rows = _read_dataset(
        output,
        "chaser_epoch_bout_histogram",
        "goodcopbadcop_export",
    )
    assert len(bout_hist_rows) == 183
    assert {
        row["metric_name"]
        for row in bout_hist_rows
    } == {
        "abs_bout_net_heading_change_deg",
        "bout_duration_s",
        "bout_heading_path_deg",
        "bout_net_heading_change_deg",
        "bout_path_length_mm",
    }
    pre_duration_hist = [
        row
        for row in bout_hist_rows
        if row["window_label"] == "pre_event" and row["metric_name"] == "bout_duration_s"
    ]
    assert pre_duration_hist
    assert pre_duration_hist[0]["epoch_behavior_component"] == "kinematics_bouts_v1"
    assert pre_duration_hist[0]["histogram_dataset"] == "per_epoch_bout_histograms"
    assert pre_duration_hist[0]["source_swim_bout_run"] == "bouts_1"
    assert sum(row["hist_count"] for row in pre_duration_hist) == 2
    assert sum(row["hist_fraction"] for row in pre_duration_hist) == 1.0
    assert "histogram_bin_contract_json" in pre_duration_hist[0]
    assert "bin_left" in pre_duration_hist[0]
    assert "bin_right" in pre_duration_hist[0]
    assert "bin_center" in pre_duration_hist[0]

    ibi_hist_rows = _read_dataset(
        output,
        "chaser_epoch_inter_bout_interval_histogram",
        "goodcopbadcop_export",
    )
    assert len(ibi_hist_rows) == 3
    assert {row["metric_name"] for row in ibi_hist_rows} == {"inter_bout_interval_s"}
    pre_ibi_hist = [row for row in ibi_hist_rows if row["window_label"] == "pre_event"]
    assert len(pre_ibi_hist) == 1
    assert pre_ibi_hist[0]["histogram_dataset"] == "per_epoch_inter_bout_interval_histograms"
    assert pre_ibi_hist[0]["source_swim_bout_run"] == "bouts_1"
    assert sum(row["hist_count"] for row in pre_ibi_hist) == 1
    assert sum(row["hist_fraction"] for row in pre_ibi_hist) == 1.0

    center_hist_rows = _read_dataset(
        output,
        "chaser_epoch_center_distance_histogram",
        "goodcopbadcop_export",
    )
    assert len(center_hist_rows) == 9
    pre_center_hist = [row for row in center_hist_rows if row["window_label"] == "pre_event"]
    assert sum(row["hist_count"] for row in pre_center_hist) == 3
    assert pre_center_hist[0]["geometry_status"] == "circle"
    assert pre_center_hist[0]["arena_radius_mm"] == 7.5

    speed_rows = _read_dataset(
        output,
        "chaser_epoch_behavior_summary",
        "goodcopbadcop_export",
    )
    assert len(speed_rows) == 3
    pre_speed = next(row for row in speed_rows if row["window_label"] == "pre_event")
    assert pre_speed["speed_sample_count"] == 3
    np.testing.assert_allclose(pre_speed["mean_speed_mm_s"], 20.0)
    assert pre_speed["valid_frame_count"] == 3
    assert pre_speed["source_track_kinematics_run"] == "tk_1"
    training_speed = next(row for row in speed_rows if row["window_label"] == "training_event")
    assert training_speed["speed_sample_count"] == 2
    np.testing.assert_allclose(training_speed["mean_speed_mm_s"], 45.0)
    np.testing.assert_allclose(training_speed["tracking_dropout_fraction"], 1.0 / 3.0)

    speed_distance_rows = _read_dataset(
        output,
        "chaser_speed_distance_bins",
        "goodcopbadcop_export",
    )
    pre_chaser_0_bin_0_speed = next(
        row
        for row in speed_distance_rows
        if row["window_label"] == "pre_event"
        and row["chaser_index"] == 0
        and row["distance_bin_index"] == 0
    )
    assert pre_chaser_0_bin_0_speed["speed_sample_count"] == 2
    np.testing.assert_allclose(pre_chaser_0_bin_0_speed["mean_speed_mm_s"], 5.0)
    np.testing.assert_allclose(pre_chaser_0_bin_0_speed["speed_sum_mm_s"], 10.0)
    assert pre_chaser_0_bin_0_speed["source_distance_path"].endswith("/distances/distance_mm")
    training_chaser_0_bin_0_speed = next(
        row
        for row in speed_distance_rows
        if row["window_label"] == "training_event"
        and row["chaser_index"] == 0
        and row["distance_bin_index"] == 0
    )
    assert training_chaser_0_bin_0_speed["speed_sample_count"] == 0
    assert training_chaser_0_bin_0_speed["mean_speed_mm_s"] is None

    histogram_rows = _read_dataset(
        output,
        "chaser_epoch_distance_histogram",
        "goodcopbadcop_export",
    )
    pre_chaser_0_bin_0 = next(
        row
        for row in histogram_rows
        if row["window_label"] == "pre_event"
        and row["chaser_index"] == 0
        and row["distance_bin_index"] == 0
    )
    assert pre_chaser_0_bin_0["bin_left_mm"] == 0.0
    assert pre_chaser_0_bin_0["bin_right_mm"] == 2.0
    assert pre_chaser_0_bin_0["bin_center_mm"] == 1.0
    assert pre_chaser_0_bin_0["hist_count"] == 1
    assert pre_chaser_0_bin_0["behavior_class_id"] == 1
    assert pre_chaser_0_bin_0["behavior_class"] == "aggressive"
    assert pre_chaser_0_bin_0["valid_sample_count"] == 3
    assert len({row["source_lineage_hash"] for row in histogram_rows}) == 18

    cra_summary_rows = _read_dataset(
        output,
        "chaser_cra_primary_endpoint_summary",
        "goodcopbadcop_export",
    )
    assert len(cra_summary_rows) == 1
    cra_summary = cra_summary_rows[0]
    assert cra_summary["export_schema_version"] == EXPORT_SCHEMA_VERSION
    assert cra_summary["table_name"] == "chaser_cra_primary_endpoint_summary"
    assert not any("benign" in key.lower() for key in cra_summary)
    assert "benign" not in {str(value).lower() for value in cra_summary.values()}
    assert cra_summary["export_run_id"] == "goodcopbadcop_export"
    assert cra_summary["cra_primary_endpoint_component"] == DEFAULT_CRA_COMPONENT_NAME
    assert cra_summary["cra_primary_endpoint_schema_id"] == "palette.goodcopbadcop.cra_primary_endpoint.v1"
    assert cra_summary["source_component_schema_id"] == "palette.goodcopbadcop.cra_primary_endpoint.v1"
    assert len(cra_summary["source_component_fingerprint"]) == 64
    assert cra_summary["source_cra_primary_endpoint_path"].endswith(
        f"/cra_primary_endpoint/{DEFAULT_CRA_COMPONENT_NAME}"
    )
    assert cra_summary["source_chaser_distance_run"] == "chaser_distance_1"
    assert cra_summary["fish_id"] == "0"
    assert cra_summary["aggressive_color"] == "#ff0000"
    assert cra_summary["inert_color"] == "#0000ff"
    assert cra_summary["pre_aggressive_quadrant"] == "top_left"
    assert cra_summary["post_aggressive_quadrant"] == "bottom_right"
    assert cra_summary["pre_inert_quadrant"] == "top_right"
    assert cra_summary["post_inert_quadrant"] == "bottom_left"
    np.testing.assert_allclose(cra_summary["delta_occ_agg"], -1.0)
    np.testing.assert_allclose(cra_summary["occ_post_inert"], 1.0)
    assert len(cra_summary["source_lineage_hash"]) == 64

    cra_object_phase_rows = _read_dataset(
        output,
        "chaser_cra_primary_endpoint_object_phase",
        "goodcopbadcop_export",
    )
    post_aggressive = next(
        row
        for row in cra_object_phase_rows
        if row["phase_label"] == "post_static" and row["object_role"] == "aggressive"
    )
    assert post_aggressive["export_run_id"] == "goodcopbadcop_export"
    assert post_aggressive["cra_primary_endpoint_component"] == DEFAULT_CRA_COMPONENT_NAME
    assert post_aggressive["source_cra_primary_endpoint_path"] == cra_summary["source_cra_primary_endpoint_path"]
    assert post_aggressive["source_component_fingerprint"] == cra_summary["source_component_fingerprint"]
    assert post_aggressive["object_index"] == 0
    assert post_aggressive["behavior_class"] == "aggressive"
    assert post_aggressive["raw_color_hex"] == "#ff0000"
    assert post_aggressive["enable_chase"] is True
    assert post_aggressive["source_window_label"] == "post_event"
    assert post_aggressive["effective_start_frame"] == 7
    assert post_aggressive["effective_end_frame"] == 8
    assert post_aggressive["settle_excluded_frame_count"] == 1
    assert post_aggressive["object_quadrant_label"] == "bottom_right"
    np.testing.assert_allclose(post_aggressive["object_x_px"], 15.0)
    np.testing.assert_allclose(post_aggressive["object_y_px"], 15.0)
    np.testing.assert_allclose(post_aggressive["occupancy_fraction"], 0.0)
    assert post_aggressive["valid_frame_count"] == 2
    assert len({row["source_lineage_hash"] for row in cra_object_phase_rows}) == 4

    cra_quadrant_rows = _read_dataset(
        output,
        "chaser_cra_quadrant_occupancy",
        "goodcopbadcop_export",
    )
    assert len(cra_quadrant_rows) == 8
    pre_chaser_quadrant = next(
        row
        for row in cra_quadrant_rows
        if row["phase_label"] == "pre_static" and row["is_chaser_quadrant"] is True
    )
    assert pre_chaser_quadrant["quadrant_id"] == "top_left"
    assert pre_chaser_quadrant["chaser_quadrant_label"] == "top_left"
    np.testing.assert_allclose(pre_chaser_quadrant["occupancy_fraction"], 1.0)
    assert pre_chaser_quadrant["quadrant_valid_frame_count"] == 3
    assert pre_chaser_quadrant["source_cra_primary_endpoint_path"] == cra_summary["source_cra_primary_endpoint_path"]
    post_chaser_quadrant = next(
        row
        for row in cra_quadrant_rows
        if row["phase_label"] == "post_static" and row["is_chaser_quadrant"] is True
    )
    assert post_chaser_quadrant["quadrant_id"] == "bottom_right"
    assert post_chaser_quadrant["chaser_quadrant_label"] == "bottom_right"
    assert post_chaser_quadrant["effective_start_frame"] == 7
    np.testing.assert_allclose(post_chaser_quadrant["occupancy_fraction"], 0.0)
    for phase_label in ("pre_static", "post_static"):
        total = sum(
            float(row["occupancy_fraction"])
            for row in cra_quadrant_rows
            if row["phase_label"] == phase_label
        )
        np.testing.assert_allclose(total, 1.0)
    assert len({row["source_lineage_hash"] for row in cra_quadrant_rows}) == 8

    near_field_summary_rows = _read_dataset(
        output,
        "chaser_cra_near_field_summary",
        "goodcopbadcop_export",
    )
    assert len(near_field_summary_rows) == 1
    near_field_summary = near_field_summary_rows[0]
    assert near_field_summary["cra_near_field_component"] == DEFAULT_CRA_NEAR_FIELD_COMPONENT_NAME
    assert near_field_summary["cra_near_field_schema_id"] == "palette.goodcopbadcop.cra_near_field.v1"
    assert near_field_summary["source_cra_primary_endpoint_path"] == cra_summary["source_cra_primary_endpoint_path"]
    assert near_field_summary["geometry_status"] == "circle"
    assert near_field_summary["arena_shape"] == "circle"
    np.testing.assert_allclose(near_field_summary["nearzone_occ_delta_agg"], -1.0)
    np.testing.assert_allclose(near_field_summary["nearzone_occ_specificity"], -1.0)
    assert len(near_field_summary["source_lineage_hash"]) == 64

    near_field_object_phase_rows = _read_dataset(
        output,
        "chaser_cra_near_field_object_phase",
        "goodcopbadcop_export",
    )
    near_field_post_aggressive = next(
        row
        for row in near_field_object_phase_rows
        if row["phase_label"] == "post_static" and row["object_role"] == "aggressive"
    )
    assert near_field_post_aggressive["export_run_id"] == "goodcopbadcop_export"
    assert near_field_post_aggressive["cra_near_field_component"] == DEFAULT_CRA_NEAR_FIELD_COMPONENT_NAME
    assert near_field_post_aggressive["source_cra_near_field_path"] == near_field_summary["source_cra_near_field_path"]
    assert near_field_post_aggressive["source_cra_primary_endpoint_path"] == cra_summary["source_cra_primary_endpoint_path"]
    assert near_field_post_aggressive["object_index"] == 0
    assert near_field_post_aggressive["raw_color_hex"] == "#ff0000"
    assert near_field_post_aggressive["phase_label"] == "post_static"
    np.testing.assert_allclose(near_field_post_aggressive["near_zone_occupancy_fraction"], 0.0)
    np.testing.assert_allclose(near_field_post_aggressive["approach_p05_mm"], float(near_field_result.approach_percentile_mm[1, 0, 0]))
    assert near_field_post_aggressive["valid_distance_count"] == 2
    assert len({row["source_lineage_hash"] for row in near_field_object_phase_rows}) == 4

    near_field_radial_rows = _read_dataset(
        output,
        "chaser_cra_near_field_radial_density",
        "goodcopbadcop_export",
    )
    assert len(near_field_radial_rows) == 12
    pre_radial_aggressive = next(
        row
        for row in near_field_radial_rows
        if row["phase_label"] == "pre_static" and row["object_role"] == "aggressive" and row["radial_bin_index"] == 0
    )
    assert pre_radial_aggressive["cra_near_field_component"] == DEFAULT_CRA_NEAR_FIELD_COMPONENT_NAME
    assert pre_radial_aggressive["source_cra_primary_endpoint_path"] == cra_summary["source_cra_primary_endpoint_path"]
    assert pre_radial_aggressive["radial_bin_left_mm"] == 0.0
    assert pre_radial_aggressive["radial_bin_right_mm"] == 2.0
    assert pre_radial_aggressive["radial_density_per_mm2"] is not None

    near_field_cdf_rows = _read_dataset(
        output,
        "chaser_cra_near_field_distance_cdf",
        "goodcopbadcop_export",
    )
    assert len(near_field_cdf_rows) == 8
    pre_cdf_aggressive = next(
        row
        for row in near_field_cdf_rows
        if row["phase_label"] == "pre_static" and row["object_role"] == "aggressive" and row["cdf_threshold_index"] == 0
    )
    assert pre_cdf_aggressive["distance_threshold_mm"] == 2.0
    assert pre_cdf_aggressive["cdf_fraction"] is not None
    assert pre_cdf_aggressive["source_cra_near_field_path"] == near_field_summary["source_cra_near_field_path"]

    egocentric_rows = _read_dataset(
        output,
        "chaser_egocentric_epoch_summary",
        "goodcopbadcop_export",
    )
    post_egocentric_chaser_1 = next(
        row
        for row in egocentric_rows
        if row["window_label"] == "post_event" and row["chaser_index"] == 1
    )
    assert post_egocentric_chaser_1["egocentric_component_name"] == "track_offline_tk_1_id_0_smoothed"
    assert post_egocentric_chaser_1["egocentric_component_path"].endswith(
        "/egocentric_bearing/track_offline_tk_1_id_0_smoothed"
    )
    assert post_egocentric_chaser_1["source_track_kinematics_run"] == "tk_1"
    assert post_egocentric_chaser_1["source_track_kinematics_scope"] == "offline"
    assert post_egocentric_chaser_1["source_track_kinematics_track_id"] == 0
    assert post_egocentric_chaser_1["source_heading_array"].endswith(
        "/tracks/id_0/smoothed_heading_degrees"
    )
    assert post_egocentric_chaser_1["heading_level"] == "smoothed"
    assert post_egocentric_chaser_1["valid_frame_count"] == int(
        egocentric_result.epoch_valid_frame_count[2, 1]
    )
    np.testing.assert_allclose(
        post_egocentric_chaser_1["mean_alignment_cos"],
        float(egocentric_result.epoch_mean_alignment_cos[2, 1]),
    )
    assert post_egocentric_chaser_1["front_definition"] == "abs(bearing_deg) <= 45"

    egocentric_histogram_rows = _read_dataset(
        output,
        "chaser_egocentric_distance_bearing_histogram",
        "goodcopbadcop_export",
    )
    pre_egocentric_chaser_0_bin = next(
        row
        for row in egocentric_histogram_rows
        if row["window_label"] == "pre_event"
        and row["chaser_index"] == 0
        and row["distance_bin_index"] == 0
        and row["bearing_bin_index"] == 0
    )
    assert pre_egocentric_chaser_0_bin["distance_bin_left_mm"] == 0.0
    assert pre_egocentric_chaser_0_bin["distance_bin_right_mm"] == 2.0
    assert pre_egocentric_chaser_0_bin["bearing_bin_left_deg"] == -180.0
    assert pre_egocentric_chaser_0_bin["bearing_bin_right_deg"] == -90.0
    assert pre_egocentric_chaser_0_bin["hist_count"] == int(
        egocentric_result.histogram_counts[0, 0, 0, 0]
    )
    np.testing.assert_allclose(
        pre_egocentric_chaser_0_bin["hist_probability"],
        float(egocentric_result.histogram_probability[0, 0, 0, 0]),
    )
    assert pre_egocentric_chaser_0_bin["valid_sample_count"] == int(
        egocentric_result.epoch_valid_frame_count[0, 0]
    )
    assert len({row["source_lineage_hash"] for row in egocentric_histogram_rows}) == 72


def test_export_cross_recording_analytics_uses_bout_kinematics_source_refs_fallback(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "recording_source_refs_analysis.zarr")
    _convert_bout_kinematics_fixture_to_compact_v2(source)
    root = zarr.open_group(str(source), mode="a")
    bout_kin = root["analysis"]["bout_kinematics_runs"]["bout_kinematics_test"]
    bout_kin.attrs["source_refs"] = {
        "source_track_id": 0,
        "source_track_kinematics_run": "tk_test",
        "source_swim_bout_run": "bouts_test",
        "source_swim_bout_speed_level": "speed_exponential",
    }
    for name in (
        "source_track_id",
        "source_track_kinematics_run",
        "source_swim_bout_run",
        "source_swim_bout_speed_level",
    ):
        del bout_kin.attrs[name]

    output = tmp_path / "exports" / "palette_analytics"
    manifest = export_sources([source], output_root=output, export_run_id="source_refs", jobs=1)

    assert manifest["row_counts_by_table"]["bout_kinematics_metrics"] == 6
    bout_kin_rows = _read_dataset(output, "bout_kinematics_metrics", "source_refs")
    assert all(row["source_swim_bout_run"] == "bouts_test" for row in bout_kin_rows)
    assert all(row["source_swim_bout_speed_level"] == "speed_exponential" for row in bout_kin_rows)
    assert all(row["source_track_kinematics_run"] == "tk_test" for row in bout_kin_rows)
    assert all(row["track_id"] == 0 for row in bout_kin_rows)


def test_export_cross_recording_analytics_reads_compact_stimulus_response(tmp_path: Path) -> None:
    source = _make_source_zarr(tmp_path / "recording_compact_response_analysis.zarr")
    _replace_stimulus_response_fixture_with_compact_v2(source)
    output = tmp_path / "exports" / "palette_analytics"

    manifest = export_sources([source], output_root=output, export_run_id="compact_response", jobs=1)

    assert manifest["row_counts_by_table"]["recording_summary"] == 1
    assert manifest["row_counts_by_table"]["stimulus_step_summary"] == 2
    assert manifest["row_counts_by_table"]["stimulus_response_per_fish_step"] == 2

    summary_rows = _read_dataset(output, "recording_summary", "compact_response")
    assert summary_rows[0]["stimulus_response_run"] == "stimulus_response_test"
    assert summary_rows[0]["global_fish_count"] == 1
    assert summary_rows[0]["total_distance_mm_sum"] == 25.0

    response_rows = _read_dataset(output, "stimulus_response_per_fish_step", "compact_response")
    moving = next(row for row in response_rows if row["stimulus_mode"] == "MOVING_GRATING")
    assert moving["omr_family"] == "moving_grating_omr"
    assert moving["omr_path_index"] == 0.75
    assert moving["first_aligned_bout_latency_s"] is None

    radial = next(row for row in response_rows if row["stimulus_mode"] == "CONCENTRIC_GRATING")
    assert radial["omr_family"] == "concentric_radial_omr"
    np.testing.assert_allclose(radial["radial_path_index"], -0.5)
    np.testing.assert_allclose(radial["first_aligned_bout_latency_s"], 0.2)


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
