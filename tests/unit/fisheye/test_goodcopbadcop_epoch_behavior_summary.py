from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_distance_io import ChaserDistanceReadError
from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.analysis.chaser_epoch_behavior_summary import (
    AUTHORITATIVE_EXECUTION_MODE,
    DEFAULT_COMPONENT_NAME,
    LEGACY_EXECUTION_MODE,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    ArenaGeometry,
    _make_per_epoch_fish,
    _require_authoritative_track_inputs,
    _require_matching_track_fps,
    _require_positive_fps,
    _resolve_speed_sources,
    _resolve_arena_geometry,
    _speed_level_key,
    _validate_result_publication_identity,
    build_chaser_epoch_behavior_summary_result as build_goodcopbadcop_epoch_behavior_summary_result,
    write_chaser_epoch_behavior_summary_component as write_goodcopbadcop_epoch_behavior_summary_component,
)
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis import chaser_epoch_behavior_summary as epoch_summary_module
from fisheye.visualization.goodcopbadcop_interactive import load_goodcopbadcop_epoch_behavior_data
from tests.unit.fisheye.goodcopbadcop_test_fixtures import (
    _add_goodcopbadcop_swim_bout_run,
)
from tests.unit.fisheye.test_cra_near_field import _add_circle_geometry
from tests.unit.fisheye.test_marimo_palette_explorer_components import (
    _make_archive_with_goodcopbadcop_egocentric_spec,
)


_DEFERRED_CHASER_SEMANTIC_AUTHORITY_REASON = (
    "chaser behavior/component/export authority is intentionally unavailable "
    "until independently sealed"
)
_REQUIRES_SEALED_CHASER_SEMANTICS = pytest.mark.xfail(
    raises=ChaserDistanceReadError,
    reason=_DEFERRED_CHASER_SEMANTIC_AUTHORITY_REASON,
    strict=True,
)


@_REQUIRES_SEALED_CHASER_SEMANTICS
def test_goodcopbadcop_epoch_behavior_summary_builds_fish_and_chaser_tables(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(
        tmp_path,
        monkeypatch,
    )
    _add_goodcopbadcop_swim_bout_run(zarr_path)
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
    assert result.source_speed_level_selection == "persisted_swim_bout_signal_level"
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


def test_epoch_behavior_rejects_detector_signal_as_physical_speed_level() -> None:
    with pytest.raises(ValueError, match="Detector-only signals"):
        _speed_level_key("exponential")


@pytest.mark.parametrize("fps", [None, 0, -1, np.nan, np.inf])
def test_epoch_behavior_rejects_missing_or_invalid_fps(fps) -> None:
    with pytest.raises(ValueError, match="finite positive fps"):
        _require_positive_fps(fps, source="fixture")


def test_authoritative_epoch_behavior_requires_explicit_speed_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    swim_tables = SimpleNamespace(
        candidate=SimpleNamespace(
            source_track_kinematics_run="tk_1",
            track_id=0,
        ),
        run_attrs={"source_track_kinematics_run": "tk_1"},
        signal=SimpleNamespace(source_level="filtered", speed_level="filtered"),
    )
    monkeypatch.setattr(
        epoch_summary_module,
        "load_default_swim_bout_tables",
        lambda *_args, **_kwargs: swim_tables,
    )

    with pytest.raises(ValueError, match="requires an explicit --speed-level"):
        _resolve_speed_sources(
            object(),
            swim_bout_run="latest",
            track_kinematics_run="tk_1",
            track_kinematics_scope="offline",
            track_id=0,
            speed_level=None,
            execution_mode=AUTHORITATIVE_EXECUTION_MODE,
        )


def test_authoritative_epoch_behavior_fails_closed_without_swim_bout_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(*_args, **_kwargs):
        raise KeyError("no bout authority")

    monkeypatch.setattr(
        epoch_summary_module,
        "load_default_swim_bout_tables",
        _missing,
    )

    with pytest.raises(ValueError, match="verified swim-bout run"):
        _resolve_speed_sources(
            object(),
            swim_bout_run="latest",
            track_kinematics_run="tk_1",
            track_kinematics_scope="offline",
            track_id=0,
            speed_level="filtered",
            execution_mode=AUTHORITATIVE_EXECUTION_MODE,
        )


def test_authoritative_epoch_behavior_requires_persisted_validity_arrays() -> None:
    track = SimpleNamespace(
        frame_indices=np.arange(3, dtype=np.int64),
        sample_valid=None,
        transition_valid=np.asarray([False, True, True]),
        speed_mm_by_level={"filtered": np.ones(3)},
        frame_path_distance_mm_by_level={"filtered": np.ones(3)},
    )
    with pytest.raises(ValueError, match="sample_valid"):
        _require_authoritative_track_inputs(
            track,
            source_speed_level="filtered",
        )


def test_authoritative_epoch_behavior_rejects_missing_track_fps() -> None:
    with pytest.raises(ValueError, match="finite positive fps"):
        _require_matching_track_fps(
            SimpleNamespace(run_attrs={}),
            fps=10.0,
        )


def test_epoch_rates_and_motion_use_valid_tracked_time_and_validity_masks(
    tmp_path,
) -> None:
    root = zarr.open_group(str(tmp_path / "epoch_hygiene.zarr"), mode="w")
    run = root.create_group("run")
    run.attrs.update({"fps": 10.0, "total_frames": 10})
    positions = run.create_group("positions")
    positions.create_array(
        "fish_valid",
        data=np.ones(10, dtype=bool),
        chunks=(10,),
    )

    sample_valid = np.asarray(
        [True, True, True, True, True, False, False, False, False, False],
        dtype=bool,
    )
    transition_valid = np.asarray(
        [False, True, True, True, True, False, False, False, False, False],
        dtype=bool,
    )
    track = SimpleNamespace(
        frame_indices=np.arange(10, dtype=np.int64),
        sample_valid=sample_valid,
        transition_valid=transition_valid,
        speed_mm_by_level={
            "filtered": np.asarray([999.0, 10.0, 10.0, 10.0, 10.0, 999.0, 999.0, 999.0, 999.0, 999.0])
        },
        frame_path_distance_mm_by_level={
            "filtered": np.asarray([999.0, 1.0, 1.0, 1.0, 1.0, 999.0, 999.0, 999.0, 999.0, 999.0])
        },
        smoothed_heading_degrees=None,
        heading_degrees=None,
    )
    bouts = np.zeros(2, dtype=[("peak_frame", np.int64)])
    bouts["peak_frame"] = [1, 4]
    intervals = np.zeros(
        1,
        dtype=[
            ("interval_s", np.float64),
            ("prev_end_frame", np.int64),
            ("next_start_frame", np.int64),
            ("valid", bool),
        ],
    )
    intervals[0] = (0.2, 1, 4, True)
    swim_tables = SimpleNamespace(
        bouts=bouts,
        inter_bout_intervals=intervals,
    )
    window = ChaserDistanceWindow(
        window_id=0,
        label="epoch",
        start_frame=0,
        end_frame=9,
        start_time_s=0.0,
        end_time_s=1.0,
        duration_s=1.0,
    )
    geometry = ArenaGeometry(
        status="missing",
        source=None,
        shape="unknown",
        width_px=np.nan,
        height_px=np.nan,
        center_x_px=None,
        center_y_px=None,
        radius_px=None,
    )

    strict = _make_per_epoch_fish(
        windows=(window,),
        run_group=run,
        swim_tables=swim_tables,
        track=track,
        source_speed_level="filtered",
        geometry=geometry,
        wall_band_mm=5.0,
        fps=10.0,
        execution_mode=AUTHORITATIVE_EXECUTION_MODE,
    )[0]
    legacy = _make_per_epoch_fish(
        windows=(window,),
        run_group=run,
        swim_tables=swim_tables,
        track=track,
        source_speed_level="filtered",
        geometry=geometry,
        wall_band_mm=5.0,
        fps=10.0,
        execution_mode=LEGACY_EXECUTION_MODE,
    )[0]

    assert int(strict["valid_tracked_frame_count"]) == 5
    assert float(strict["valid_tracked_duration_s"]) == 0.5
    assert int(strict["motion_valid_sample_count"]) == 4
    assert int(strict["speed_sample_count"]) == 4
    assert float(strict["mean_speed_mm_s"]) == 10.0
    assert float(strict["total_path_mm"]) == 4.0
    assert float(strict["bout_rate_denominator_s"]) == 0.5
    assert float(strict["bout_rate_per_min"]) == 240.0
    assert float(strict["inter_bout_interval_rate_per_min"]) == 120.0
    assert strict["bout_rate_denominator"].rstrip(b"\x00") == b"valid_tracked_duration_s"
    assert strict["wall_fraction_denominator"].rstrip(b"\x00") == b"valid_in_arena_center_samples"

    assert "valid_tracked_duration_s" not in (legacy.dtype.names or ())
    assert float(legacy["bout_rate_per_min"]) == 120.0
    assert int(legacy["speed_sample_count"]) == 10


def test_authoritative_publication_rejects_missing_source_warning() -> None:
    required_fields = [
        ("valid_tracked_frame_count", np.int64),
        ("valid_tracked_duration_s", np.float64),
        ("valid_tracked_duration_source", "S64"),
        ("motion_valid_sample_count", np.int64),
        ("motion_validity_rule", "S64"),
        ("wall_fraction_denominator_count", np.int64),
        ("wall_fraction_denominator", "S64"),
        ("bout_rate_denominator_s", np.float64),
        ("bout_rate_denominator", "S64"),
        ("inter_bout_interval_rate_denominator_s", np.float64),
        ("inter_bout_interval_rate_denominator", "S64"),
    ]
    result = SimpleNamespace(
        execution_mode=AUTHORITATIVE_EXECUTION_MODE,
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        method_version=METHOD_VERSION,
        source_track_kinematics_run="tk_1",
        source_track_kinematics_scope="offline",
        source_track_kinematics_track_id=0,
        source_track_kinematics_track_path=(
            "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0"
        ),
        source_speed_level="filtered",
        source_speed_level_selection="explicit_physical_track_speed_level",
        source_swim_bout_run="bouts_1",
        source_swim_bout_path="analysis/swim_bout_runs/bouts_1",
        source_swim_bout_level_path="analysis/swim_bout_runs/bouts_1/speed_filtered",
        fps=10.0,
        warnings=("track_kinematics_unavailable: tampered",),
        per_epoch_fish=np.zeros(1, dtype=required_fields),
    )

    with pytest.raises(ValueError, match="cannot complete with missing"):
        _validate_result_publication_identity(
            result,
            expected_mode=AUTHORITATIVE_EXECUTION_MODE,
            expected_schema_id=SCHEMA_ID,
            expected_schema_version=SCHEMA_VERSION,
            expected_method_version=METHOD_VERSION,
        )


@pytest.mark.parametrize("pixels_per_mm", [None, 0.0, -1.0])
def test_epoch_behavior_rejects_missing_or_nonpositive_arena_scale(
    tmp_path,
    pixels_per_mm,
) -> None:
    root = zarr.open_group(str(tmp_path / "missing_scale.zarr"), mode="w")
    run = root.require_group("analysis/chaser_distance_runs/run_1")
    if pixels_per_mm is not None:
        run.attrs["pixels_per_mm_projector"] = pixels_per_mm

    with pytest.raises(ValueError, match="refusing the historical 1.0 fallback"):
        _resolve_arena_geometry(root, run)


@_REQUIRES_SEALED_CHASER_SEMANTICS
def test_goodcopbadcop_epoch_behavior_summary_writes_and_reads_component(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(
        tmp_path,
        monkeypatch,
    )
    _add_goodcopbadcop_swim_bout_run(zarr_path)
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
