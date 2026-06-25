from __future__ import annotations

from pathlib import Path

from PIL import Image

from fisheye.utils.export_goodcopbadcop_static_montage import (
    LoadedTile,
    SourceRecording,
    _compose_page,
    _compose_recording_panel,
    _recording_id_from_zarr_path,
    default_goodcopbadcop_artifact_specs,
)


def test_default_goodcopbadcop_artifact_specs_are_curated_png_paths() -> None:
    specs = default_goodcopbadcop_artifact_specs()

    assert len(specs) == 11
    assert [spec.artifact_id for spec in specs] == [
        "detection_occupancy_overview",
        "chaser_distance_timeseries",
        "chaser_distance_epoch_median",
        "chaser_distance_epoch_distribution",
        "cra_primary_endpoint_overview",
        "cra_near_field_summary",
        "cra_near_field_radial_density",
        "cra_near_field_distance_cdf",
        "egocentric_bearing_polar",
        "egocentric_bearing_point_cloud",
        "track_kinematics_summary",
    ]
    assert all(not spec.path.startswith("/") for spec in specs)
    assert all("/visualizations/" in spec.path for spec in specs)
    assert all(spec.path.endswith("_png") for spec in specs)


def test_recording_id_from_zarr_path_strips_analysis_suffix() -> None:
    path = Path("/tmp/2026-06-21T18-18-31Z_arena_1_GoodCopBadCop_analysis.zarr")

    assert _recording_id_from_zarr_path(path) == "2026-06-21T18-18-31Z_arena_1_GoodCopBadCop"


def test_compose_recording_panel_and_page_dimensions() -> None:
    source = SourceRecording(recording_id="rec_001", zarr_path=Path("/tmp/rec_001_analysis.zarr"))
    tiles = [
        LoadedTile(
            artifact_id="present",
            label="Present plot",
            path="analysis/example/visualizations/present_png",
            image=Image.new("RGB", (100, 50), (255, 0, 0)),
            error=None,
        ),
        LoadedTile(
            artifact_id="missing",
            label="Missing plot",
            path="analysis/example/visualizations/missing_png",
            image=None,
            error="not found",
        ),
    ]

    panel = _compose_recording_panel(
        source=source,
        tiles=tiles,
        columns=2,
        tile_width=120,
        max_image_height=80,
        margin=10,
        gutter=5,
    )

    assert panel.size == (265, 190)

    page = _compose_page(
        export_run_id="export_001",
        page_index=0,
        page_count=1,
        page_sources=[source],
        panels=[panel],
        margin=10,
        gutter=5,
    )

    assert page.size == (285, 282)
