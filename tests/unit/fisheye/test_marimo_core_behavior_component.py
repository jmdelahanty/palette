from __future__ import annotations

import polars as pl

import apps.marimo.components.core_behavior as core_component
from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.core_behavior import (
    CoreBehaviorSource,
    collect_projection,
    scan_export_parquet,
)
from apps.marimo.components.registry import (
    discover_interactive_spec_options,
    supported_renderer_ids,
)
from tests.unit.fisheye.test_interactive_track_kinematics import (
    _add_eye_angle_run,
    _add_hierarchical_swim_bouts,
    _make_archive_with_interactive_artifact,
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


def test_core_source_exposes_only_lineage_compatible_swim_bouts(tmp_path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(zarr_path)
    source = CoreBehaviorSource(zarr_path, _core_option(zarr_path))

    assert "swim_bouts" in source.available_analysis_ids()
    projection = source.project_swim_bouts()

    assert isinstance(projection.frame, pl.LazyFrame)
    assert projection.row_count == 2
    assert {"start_s", "end_s", "duration_s"}.issubset(projection.columns)


def test_core_source_exposes_eye_angles_only_when_persisted(tmp_path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    source_without_eyes = CoreBehaviorSource(zarr_path, _core_option(zarr_path))
    assert "eye_angles" not in source_without_eyes.available_analysis_ids()

    _add_eye_angle_run(zarr_path)
    source = CoreBehaviorSource(zarr_path, _core_option(zarr_path))
    projection = source.project_eye_angles(start_s=0.0, stop_s=0.01)

    assert "eye_angles" in source.available_analysis_ids()
    assert isinstance(projection.frame, pl.LazyFrame)
    assert projection.row_count == 3
    assert "mean_eye_vergence_gaze_deg_smoothed" in projection.columns


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
