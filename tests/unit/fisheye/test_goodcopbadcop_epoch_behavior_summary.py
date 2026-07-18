from __future__ import annotations

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.analysis.chaser_epoch_behavior_summary import (
    DEFAULT_COMPONENT_NAME,
    SCHEMA_ID,
    build_chaser_epoch_behavior_summary_result as build_goodcopbadcop_epoch_behavior_summary_result,
    write_chaser_epoch_behavior_summary_component as write_goodcopbadcop_epoch_behavior_summary_component,
)
from fisheye.visualization.goodcopbadcop_interactive import load_goodcopbadcop_epoch_behavior_data
from tests.unit.fisheye.test_marimo_palette_explorer_components import (
    _add_swim_bout_run,
    _make_archive_with_goodcopbadcop_egocentric_spec,
)
from tests.unit.fisheye.test_cra_near_field import _add_circle_geometry


def test_goodcopbadcop_epoch_behavior_summary_builds_fish_and_chaser_tables(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    _add_swim_bout_run(zarr_path)
    _add_circle_geometry(zarr_path)

    result = build_goodcopbadcop_epoch_behavior_summary_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="latest",
    )

    assert result.component_name == DEFAULT_COMPONENT_NAME
    assert result.chaser_distance_run_path == "analysis/chaser_distance_runs/chaser_distance_1"
    assert result.source_swim_bout_run == "bouts_1"
    assert result.source_track_kinematics_run == "tk_1"
    assert result.source_speed_level == "filtered"
    assert result.per_epoch_fish.shape == (3,)
    assert result.per_epoch_chaser.shape == (6,)
    assert result.per_epoch_bouts.shape == (4,)
    assert result.per_epoch_bout_histograms.shape[0] > 0
    assert result.per_epoch_inter_bout_interval_histograms.shape[0] > 0
    assert result.center_distance_histogram.shape == (9,)
    assert result.arena_geometry.status == "circle"

    labels = [
        value.decode("utf-8").rstrip("\x00")
        for value in result.per_epoch_fish["window_label"]
    ]
    assert labels == ["pre_event", "training_event", "post_event"]

    pre = result.per_epoch_fish[0]
    assert int(pre["bout_count"]) == 2
    assert float(pre["mean_bout_duration_s"]) == 0.05
    assert float(pre["mean_bout_path_length_mm"]) == 0.25
    assert "mean_bout_net_heading_change_deg" in result.per_epoch_fish.dtype.names
    assert "mean_abs_bout_net_heading_change_deg" in result.per_epoch_fish.dtype.names
    assert "wall_fraction" in result.per_epoch_fish.dtype.names
    assert int(pre["inter_bout_interval_count"]) == 1
    assert float(pre["mean_inter_bout_interval_s"]) == 0.06
    assert float(pre["median_inter_bout_interval_s"]) == 0.06
    assert int(pre["speed_sample_count"]) == 3
    assert float(pre["mean_speed_mm_s"]) == 20.0
    assert np.isfinite(float(pre["tracking_dropout_fraction"]))
    pre_bouts = result.per_epoch_bouts[result.per_epoch_bouts["window_label"] == b"pre_event"]
    assert pre_bouts.shape == (2,)
    assert "bout_duration_s" in result.per_epoch_bouts.dtype.names
    assert "bout_path_length_mm" in result.per_epoch_bouts.dtype.names
    assert "bout_net_heading_change_deg" in result.per_epoch_bouts.dtype.names
    np.testing.assert_allclose(pre_bouts["bout_duration_s"], [0.04, 0.06])
    np.testing.assert_allclose(pre_bouts["bout_path_length_mm"], [0.2, 0.3])
    pre_duration_hist = result.per_epoch_bout_histograms[
        (result.per_epoch_bout_histograms["window_label"] == b"pre_event")
        & (result.per_epoch_bout_histograms["metric_name"] == b"bout_duration_s")
    ]
    assert int(np.sum(pre_duration_hist["hist_count"])) == 2
    pre_ibi_hist = result.per_epoch_inter_bout_interval_histograms[
        result.per_epoch_inter_bout_interval_histograms["window_label"] == b"pre_event"
    ]
    assert int(np.sum(pre_ibi_hist["hist_count"])) == 1

    pre_chaser_0 = result.per_epoch_chaser[
        (result.per_epoch_chaser["window_label"] == b"pre_event")
        & (result.per_epoch_chaser["chaser_index"] == 0)
    ][0]
    assert int(pre_chaser_0["distance_sample_count"]) > 0
    assert np.isfinite(float(pre_chaser_0["median_distance_mm"]))


def test_goodcopbadcop_epoch_behavior_summary_writes_and_reads_component(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    _add_swim_bout_run(zarr_path)
    _add_circle_geometry(zarr_path)
    result = build_goodcopbadcop_epoch_behavior_summary_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
    )

    component_path = write_goodcopbadcop_epoch_behavior_summary_component(
        zarr_path,
        result,
        overwrite=True,
    )

    assert component_path.endswith("/epoch_behavior_summary/kinematics_bouts_v1")
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["analysis/chaser_distance_runs/chaser_distance_1/epoch_behavior_summary"]
    assert parent.attrs["latest"] == "kinematics_bouts_v1"
    assert parent.attrs["latest_complete"] == "kinematics_bouts_v1"
    component = root[component_path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["status"] == "complete"
    assert component.attrs["source_refs"]["source_swim_bout_run"] == "bouts_1"
    assert component.attrs["source_refs"]["source_track_kinematics_run"] == "tk_1"

    stored_fish, fish_attrs = load_structured_dataset(component, "per_epoch_fish")
    stored_chaser, chaser_attrs = load_structured_dataset(component, "per_epoch_chaser")
    stored_bouts, bout_attrs = load_structured_dataset(component, "per_epoch_bouts")
    stored_bout_hist, bout_hist_attrs = load_structured_dataset(component, "per_epoch_bout_histograms")
    stored_ibi_hist, ibi_hist_attrs = load_structured_dataset(component, "per_epoch_inter_bout_interval_histograms")
    stored_center_hist, center_hist_attrs = load_structured_dataset(component, "center_distance_histogram")
    assert fish_attrs["row_axis"] == "stimulus_epoch_windows"
    assert chaser_attrs["row_axis"] == "stimulus_epoch_windows_x_chasers"
    assert bout_attrs["row_axis"] == "stimulus_epoch_windows_x_swim_bouts"
    assert bout_attrs["unit_of_analysis"] == "swim_bout"
    assert bout_hist_attrs["row_axis"] == "stimulus_epoch_windows_x_bout_metrics_x_bins"
    assert bout_hist_attrs["bin_contract"] == "analysis_owned_shared_bins_per_metric_within_component"
    assert ibi_hist_attrs["row_axis"] == "stimulus_epoch_windows_x_inter_bout_interval_bins"
    assert center_hist_attrs["row_axis"] == "stimulus_epoch_windows_x_center_distance_bins"
    np.testing.assert_array_equal(stored_fish["bout_count"], result.per_epoch_fish["bout_count"])
    np.testing.assert_allclose(
        stored_fish["mean_inter_bout_interval_s"],
        result.per_epoch_fish["mean_inter_bout_interval_s"],
    )
    np.testing.assert_array_equal(stored_chaser["chaser_index"], result.per_epoch_chaser["chaser_index"])
    np.testing.assert_array_equal(stored_bouts["bout_source_row"], result.per_epoch_bouts["bout_source_row"])
    np.testing.assert_array_equal(stored_bout_hist["hist_count"], result.per_epoch_bout_histograms["hist_count"])
    np.testing.assert_array_equal(stored_ibi_hist["hist_count"], result.per_epoch_inter_bout_interval_histograms["hist_count"])
    np.testing.assert_array_equal(stored_center_hist["hist_count"], result.center_distance_histogram["hist_count"])

    loaded = load_goodcopbadcop_epoch_behavior_data(
        zarr_path,
        run_path="analysis/chaser_distance_runs/chaser_distance_1",
    )
    assert loaded is not None
    assert loaded.component_path == component_path
    assert loaded.per_epoch_fish_df.height == 3
    assert loaded.per_epoch_chaser_df.height == 6
    assert loaded.per_epoch_bouts_df.height == 4
    assert loaded.per_epoch_bout_histograms_df.height == result.per_epoch_bout_histograms.shape[0]
    assert loaded.per_epoch_inter_bout_interval_histograms_df.height == result.per_epoch_inter_bout_interval_histograms.shape[0]
    assert loaded.center_distance_histogram_df.height == 9
    pre = loaded.per_epoch_fish_df.filter(loaded.per_epoch_fish_df["window_label"] == "pre_event").row(
        0,
        named=True,
    )
    assert pre["bout_count"] == 2
    assert pre["mean_inter_bout_interval_s"] == 0.06
    assert pre["mean_bout_duration_s"] == 0.05
