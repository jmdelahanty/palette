from __future__ import annotations

import numpy as np
import zarr

from fisheye.analysis.stimulus_response_io import (
    concentric_radial_omr_steps,
    moving_grating_omr_steps,
    resolve_stimulus_response_tables,
)
from fisheye.analysis.stimulus_response import (
    GratingStepData,
    ProtocolStep,
    STIMULUS_RESPONSE_LAYOUT_COMPACT_V2,
    write_stimulus_response_run,
)
from fisheye.analysis.stimulus_response_omr import OMRStepData
from tests.unit.fisheye.test_stimulus_response import (
    _synthetic_coordinate_authority,
    _synthetic_track_motion_lineage,
)


def _array(group, name: str, values) -> None:
    group.create_array(name, data=np.asarray(values), overwrite=True)


def _make_response_run() -> zarr.Group:
    root = zarr.group()
    run = root.create_group("stimulus_response")
    run.attrs.update(
        {
            "source_stimulus_run": "stimulus_test",
            "source_track_kinematics_run": "tk_test",
            "source_bout_run": "bouts_test",
        }
    )
    global_group = run.create_group("global")
    _array(global_group, "fish_id", [0])
    _array(global_group, "total_distance_mm", [12.5])
    global_omr = global_group.create_group("omr").create_group("per_fish")
    _array(global_omr, "fish_id", [0])
    _array(global_omr, "omr_path_index_weighted_by_path", [0.75])

    frames = run.create_group("frames")
    _array(frames, "step_index", [0, 0, 1, 1])

    steps = run.create_group("steps")
    step0 = steps.create_group("step_0")
    step0.attrs.update(
        {
            "step_index": 0,
            "step_name": "moving",
            "stimulus_mode": "MOVING_GRATING",
            "stimulus_mode_id": 3,
            "start_frame": 10,
            "end_frame": 20,
            "duration_s": 1.0,
        }
    )
    per_fish0 = step0.create_group("per_fish")
    _array(per_fish0, "fish_id", [0])
    _array(per_fish0, "total_distance_mm", [5.0])
    grating = step0.create_group("grating")
    grating_pf = grating.create_group("per_fish")
    _array(grating_pf, "fish_id", [0])
    _array(grating_pf, "mean_alignment_cos", [0.9])
    omr = grating.create_group("omr")
    omr.attrs["stimulus_direction_deg"] = 0.0
    omr_pf = omr.create_group("per_fish")
    _array(omr_pf, "fish_id", [0])
    _array(omr_pf, "omr_path_index", [0.7])
    omr_bouts = omr.create_group("per_bout")
    _array(omr_bouts, "bout_id", [2])
    _array(omr_bouts, "per_bout_omr_score", [0.6])
    omr_windows = omr.create_group("windows")
    _array(omr_windows, "window_length_s", [5.0])
    _array(omr_windows, "omr_path_index", [0.4])

    step1 = steps.create_group("step_1")
    step1.attrs.update(
        {
            "step_index": 1,
            "step_name": "concentric",
            "stimulus_mode": "CONCENTRIC_GRATING",
            "stimulus_mode_id": 6,
            "start_frame": 20,
            "end_frame": 30,
            "duration_s": 1.0,
        }
    )
    per_fish1 = step1.create_group("per_fish")
    _array(per_fish1, "fish_id", [0])
    _array(per_fish1, "total_distance_mm", [7.5])
    concentric = step1.create_group("concentric_grating")
    radial = concentric.create_group("radial_omr")
    radial.attrs["stimulus_radial_polarity"] = "expanding"
    radial_pf = radial.create_group("per_fish")
    _array(radial_pf, "fish_id", [0])
    _array(radial_pf, "radial_path_index", [0.8])
    radial_early = radial.create_group("early_windows")
    _array(radial_early, "window_length_s", [1.0])
    _array(radial_early, "omr_path_index", [0.5])

    return run


def test_resolve_stimulus_response_tables_reads_hierarchical_v1() -> None:
    run = _make_response_run()

    tables = resolve_stimulus_response_tables(run)

    assert tables.layout == "hierarchical_v1"
    assert tables.attrs["source_stimulus_run"] == "stimulus_test"
    assert tables.global_per_fish["total_distance_mm"].tolist() == [12.5]
    assert tables.global_omr_per_fish["omr_path_index_weighted_by_path"].tolist() == [0.75]
    assert tables.frame_annotations["step_index"].tolist() == [0, 0, 1, 1]
    assert [step.step_index for step in tables.steps] == [0, 1]
    np.testing.assert_allclose(tables.steps[0].grating_per_fish["mean_alignment_cos"], [0.9])
    assert tables.steps[0].moving_grating_omr is not None
    assert tables.steps[0].moving_grating_omr.per_bout["bout_id"].tolist() == [2]
    assert tables.steps[0].moving_grating_omr.windows["omr_path_index"].tolist() == [0.4]
    assert tables.steps[1].concentric_radial_omr is not None
    assert tables.steps[1].concentric_radial_omr.attrs["stimulus_radial_polarity"] == "expanding"
    assert tables.steps[1].concentric_radial_omr.early_windows["omr_path_index"].tolist() == [0.5]


def test_stimulus_response_omr_step_helpers_filter_families() -> None:
    run = _make_response_run()

    moving = moving_grating_omr_steps(run)
    radial = concentric_radial_omr_steps(run)

    assert [step.step_name for step in moving] == ["moving"]
    assert moving[0].moving_grating_omr is not None
    assert moving[0].moving_grating_omr.per_fish["omr_path_index"].tolist() == [0.7]
    assert [step.step_name for step in radial] == ["concentric"]
    assert radial[0].concentric_radial_omr is not None
    assert radial[0].concentric_radial_omr.per_fish["radial_path_index"].tolist() == [0.8]


def test_compact_writer_roundtrips_through_resolver() -> None:
    root = zarr.group()
    steps = [
        ProtocolStep(0, "moving", "MOVING_GRATING", 3, 10, 20, 1.0),
        ProtocolStep(1, "baseline", "SOLID_BLACK", 4, 20, 30, 1.0),
    ]
    global_metrics = {
        "fish_id": np.asarray([0], dtype=np.int32),
        "total_distance_mm": np.asarray([12.5], dtype=np.float32),
        "mean_speed_mm_s": np.asarray([3.0], dtype=np.float32),
        "fraction_moving": np.asarray([0.5], dtype=np.float32),
        "total_active_s": np.asarray([1.0], dtype=np.float32),
    }
    step_metrics = [
        {
            "fish_id": np.asarray([0], dtype=np.int32),
            "total_distance_mm": np.asarray([5.0], dtype=np.float32),
            "mean_speed_mm_s": np.asarray([4.0], dtype=np.float32),
        },
        {
            "fish_id": np.asarray([0], dtype=np.int32),
            "total_distance_mm": np.asarray([7.5], dtype=np.float32),
            "mean_speed_mm_s": np.asarray([2.0], dtype=np.float32),
        },
    ]
    omr = OMRStepData(
        per_fish={
            "fish_id": np.asarray([0], dtype=np.int32),
            "omr_path_index": np.asarray([0.75], dtype=np.float32),
        },
        per_bout={
            "fish_id": np.asarray([0], dtype=np.int32),
            "bout_id": np.asarray([4], dtype=np.int32),
            "per_bout_omr_score": np.asarray([0.6], dtype=np.float32),
        },
        windows={
            "fish_id": np.asarray([0], dtype=np.int32),
            "window_length_s": np.asarray([5.0], dtype=np.float32),
            "omr_path_index": np.asarray([0.4], dtype=np.float32),
        },
        early_windows={},
        attrs={"stimulus_direction_deg": 0.0, "method_version": "test_omr"},
    )
    grating = GratingStepData(
        per_frame={},
        per_fish={
            "fish_id": np.asarray([0], dtype=np.int32),
            "mean_alignment_cos": np.asarray([0.9], dtype=np.float32),
        },
        time_series={},
        omr=omr,
    )

    run_name = write_stimulus_response_run(
        root,
        global_metrics=global_metrics,
        steps=steps,
        step_metrics=step_metrics,
        frame_annotations={
            "step_index": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "stimulus_mode_id": np.asarray([3, 3, 4, 4], dtype=np.int32),
        },
        step_grating_data={0: grating},
        source_kinematics_run="tk",
        source_kinematics_type="offline",
        source_stimulus_run="stim",
        upstream_lineage=_synthetic_track_motion_lineage(
            run_name="tk",
            scope="offline",
            track_ids=(0,),
            fps=30.0,
        ),
        coordinate_authority=_synthetic_coordinate_authority("stim"),
        parameters={
            "layout": STIMULUS_RESPONSE_LAYOUT_COMPACT_V2,
            "fps": 30.0,
        },
        run_name="compact",
        layout=STIMULUS_RESPONSE_LAYOUT_COMPACT_V2,
    )

    run = root["analysis"]["stimulus_response_runs"][run_name]
    assert run.attrs["layout"] == STIMULUS_RESPONSE_LAYOUT_COMPACT_V2
    assert "steps" not in run
    assert "step_index" in run
    assert "step_per_fish" in run
    assert "moving_grating_omr_per_fish" in run

    tables = resolve_stimulus_response_tables(run)
    assert tables.layout == STIMULUS_RESPONSE_LAYOUT_COMPACT_V2
    assert tables.global_per_fish["total_distance_mm"].tolist() == [12.5]
    assert tables.frame_annotations["stimulus_mode_id"].tolist() == [3, 3, 4, 4]
    assert [step.step_index for step in tables.steps] == [0, 1]
    assert tables.steps[0].per_fish["total_distance_mm"].tolist() == [5.0]
    np.testing.assert_allclose(tables.steps[0].grating_per_fish["mean_alignment_cos"], [0.9])
    assert tables.steps[0].moving_grating_omr is not None
    assert tables.steps[0].moving_grating_omr.attrs["method_version"] == "test_omr"
    np.testing.assert_allclose(tables.steps[0].moving_grating_omr.per_fish["omr_path_index"], [0.75])
    assert tables.steps[0].moving_grating_omr.per_bout["bout_id"].tolist() == [4]
    np.testing.assert_allclose(tables.steps[0].moving_grating_omr.windows["omr_path_index"], [0.4])
