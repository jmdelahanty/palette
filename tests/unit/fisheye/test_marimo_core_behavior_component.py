from __future__ import annotations

import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

import apps.marimo.components.core_behavior as core_component
from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.core_behavior import (
    CoreBehaviorSource,
    CoreBehaviorProjection,
    build_core_behavior_output,
    collect_projection,
    discover_core_behavior_options,
    scan_export_parquet,
)
from apps.marimo.components.registry import (
    discover_interactive_spec_options,
    discover_recording_explorer_spec_options,
    supported_renderer_ids,
)
from tests.unit.fisheye.test_interactive_track_kinematics import (
    _add_eye_angle_run,
    _add_hierarchical_swim_bouts,
    _make_archive_with_interactive_artifact,
)
from tests.unit.fisheye.test_plot_track_kinematics_artifacts import (
    _make_track_kinematics_archive,
)


def _core_option(zarr_path):
    return next(
        option
        for option in discover_interactive_spec_options(zarr_path)
        if option.renderer == core_component.TRACK_KINEMATICS_RENDERER
    )


def test_track_renderer_is_registered_and_routes_to_core_provider(tmp_path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    option = _core_option(zarr_path)

    assert option.is_supported is True
    assert option.renderer in supported_renderer_ids()
    assert group_specs_by_provider([option]) == {"core_behavior": [option]}

    targeted = discover_recording_explorer_spec_options(zarr_path)
    assert [item.artifact_path for item in targeted] == [option.artifact_path]


def test_canonical_track_run_is_discovered_without_visualization_spec(tmp_path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)

    assert discover_recording_explorer_spec_options(zarr_path) == []
    options = discover_core_behavior_options(zarr_path)

    assert len(options) == 1
    assert options[0].run_path == "analysis/track_kinematics_runs/offline/track_kinematics_1"
    assert options[0].track_id == 0
    assert options[0].interactive_option is None
    source = CoreBehaviorSource(zarr_path, options[0])
    assert {"speed", "heading", "position"}.issubset(source.available_analysis_ids())


def test_core_source_defers_zarr_open_until_projection(tmp_path, monkeypatch) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    option = _core_option(zarr_path)
    real_open = core_component.open_zarr_root
    calls = []

    def _tracked_open(*args, **kwargs):
        calls.append((args, kwargs))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(core_component, "open_zarr_root", _tracked_open)
    source = CoreBehaviorSource(zarr_path, option)

    assert calls == []
    assert "speed_smoothed_mm" in source.series_for("speed")
    assert calls == []

    projection = source.project_timeseries("speed", start_s=0.005, stop_s=0.015)

    assert len(calls) == 1
    assert isinstance(projection.frame, pl.LazyFrame)
    assert projection.row_count == 2
    assert "speed_smoothed_mm" in projection.columns
    collected = collect_projection(
        projection,
        columns=("time_s", "speed_smoothed_mm"),
        start_s=0.01,
    )
    assert collected.columns == ["time_s", "speed_smoothed_mm"]
    assert collected.height == 1


def test_core_source_exposes_only_lineage_compatible_swim_bouts(
    tmp_path,
    monkeypatch,
) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(zarr_path)
    source = CoreBehaviorSource(zarr_path, _core_option(zarr_path))

    assert "swim_bouts" in source.available_analysis_ids()
    projection = source.project_swim_bouts()

    assert isinstance(projection.frame, pl.LazyFrame)
    assert projection.row_count == 2
    assert {"start_s", "end_s", "duration_s"}.issubset(projection.columns)
    assert "speed_trace" in projection.related_frames
    speed_trace = projection.related_frames["speed_trace"].collect()
    assert speed_trace.columns == ["time_s", "speed_smoothed_mm"]

    class _Ui:
        @staticmethod
        def table(frame, selection=None, page_size=10):
            return frame

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return text

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def hstack(items):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

    def _forbid_to_pandas(*args, **kwargs):
        raise AssertionError("core plotting must not require Polars.to_pandas")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", _forbid_to_pandas)
    output = build_core_behavior_output(_Mo, go, px, projection=projection)
    segmentation_figure = output[2][0]
    assert [trace.type for trace in segmentation_figure.data] == ["scattergl", "bar"]
    assert segmentation_figure.data[1].name == "Persisted swim bouts"
    assert segmentation_figure.layout.yaxis2.overlaying == "y"
    distribution_figures = output[2][2]
    assert [figure.layout.title.text for figure in distribution_figures] == [
        "Swim-bout duration distribution",
        "Swim-bout distance distribution",
        "Swim-bout mean-speed distribution",
    ]
    assert [figure.layout.xaxis.title.text for figure in distribution_figures] == [
        "Duration (s)",
        "Distance (mm)",
        "Mean speed (mm/s)",
    ]

    bounded = source.project_swim_bouts(start_s=0.0, stop_s=0.03)
    assert bounded.row_count == 1
    assert bounded.related_frames["speed_trace"].collect().height == 6


def test_core_source_exposes_eye_angles_only_when_persisted(
    tmp_path,
    monkeypatch,
) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    source_without_eyes = CoreBehaviorSource(zarr_path, _core_option(zarr_path))
    assert "eye_angles" not in source_without_eyes.available_analysis_ids()

    _add_eye_angle_run(zarr_path)
    source = CoreBehaviorSource(zarr_path, _core_option(zarr_path))

    def _forbid_from_pandas(*args, **kwargs):
        raise AssertionError("eye projection must not require Polars.from_pandas")

    monkeypatch.setattr(pl, "from_pandas", _forbid_from_pandas)
    projection = source.project_eye_angles(
        run_name="latest",
        representation="nasal_gaze",
        start_s=0.0,
        stop_s=0.01,
        series_keys=("mean_eye_vergence_gaze_deg_smoothed",),
    )

    assert "eye_angles" in source.available_analysis_ids()
    assert isinstance(projection.frame, pl.LazyFrame)
    assert projection.row_count == 3
    assert "mean_eye_vergence_gaze_deg_smoothed" in projection.columns
    assert projection.metadata["eye_run_name"] == "eye_angle_1"
    assert projection.metadata["representation"] == "nasal_gaze"
    assert projection.metadata["persisted_pngs"] == ()


def test_position_plot_avoids_arrow_backed_pandas_bridge(monkeypatch) -> None:
    frame = pl.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        }
    )
    projection = CoreBehaviorProjection(
        analysis_id="position",
        frame=frame.lazy(),
        columns=tuple(frame.columns),
        source_paths=("test",),
        start_s=0.0,
        stop_s=0.2,
        row_count=frame.height,
        load_duration_ms=0.0,
        note="test",
    )

    class _Mo:
        @staticmethod
        def md(text):
            return text

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def hstack(items):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

    def _forbid_to_pandas(*args, **kwargs):
        raise AssertionError("core plotting must not require Polars.to_pandas")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", _forbid_to_pandas)
    figure = build_core_behavior_output(_Mo, go, px, projection=projection)[2]

    assert figure.data[0].type == "scattergl"
    assert list(figure.data[0].x) == [1.0, 2.0, 3.0]


def test_export_parquet_uses_true_polars_lazy_scan(tmp_path) -> None:
    parquet_path = tmp_path / "baseline_behavior_time_bins.parquet"
    pl.DataFrame(
        {
            "recording_id": ["a", "a", "b"],
            "time_bin": [0, 1, 0],
            "speed_mm_s": [1.0, 2.0, 3.0],
        }
    ).write_parquet(parquet_path)

    lazy = scan_export_parquet(
        parquet_path,
        columns=("recording_id", "speed_mm_s"),
    )

    assert isinstance(lazy, pl.LazyFrame)
    result = lazy.filter(pl.col("recording_id") == "a").collect()
    assert result.columns == ["recording_id", "speed_mm_s"]
    assert result["speed_mm_s"].to_list() == [1.0, 2.0]


def test_dense_core_plot_enforces_serialized_point_budget() -> None:
    row_count = 20000
    series_count = 10
    frame = pl.DataFrame(
        {
            "time_s": np.arange(row_count, dtype=np.float64) / 100.0,
            **{
                f"speed_series_{index}_mm": np.full(row_count, float(index), dtype=np.float32)
                for index in range(series_count)
            },
        }
    )
    projection = CoreBehaviorProjection(
        analysis_id="speed",
        frame=frame.lazy(),
        columns=tuple(frame.columns),
        source_paths=("test",),
        start_s=0.0,
        stop_s=200.0,
        row_count=row_count,
        load_duration_ms=0.0,
        note="test",
    )

    class _Ui:
        @staticmethod
        def table(frame, selection=None, page_size=10):
            return frame

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return text

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def hstack(items):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

    figure = build_core_behavior_output(_Mo, go, px, projection=projection)[2]

    assert len(figure.data) == series_count
    assert sum(len(trace.x) for trace in figure.data) <= 60000
