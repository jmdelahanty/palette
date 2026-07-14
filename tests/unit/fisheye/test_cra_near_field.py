from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.cra_near_field import (
    COMPONENT_PARENT_NAME,
    DEFAULT_COMPONENT_NAME,
    DISTANCE_CDF_PNG_ARTIFACT_NAME,
    INTERACTIVE_ARTIFACT_NAME,
    RADIAL_DENSITY_PNG_ARTIFACT_NAME,
    SCHEMA_ID,
    SUMMARY_PNG_ARTIFACT_NAME,
    ArenaGeometry,
    _available_annulus_area_mm2,
    build_cra_near_field_result,
    compute_hysteresis_visits,
    write_cra_near_field_component,
)
from fisheye.analysis.cra_primary_endpoint import (
    build_cra_primary_endpoint_result,
    write_cra_primary_endpoint_component,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID
from tests.unit.fisheye.test_cra_primary_endpoint import _make_archive, _make_chaser_result


def _decode_first(array: zarr.Array) -> str:
    return decode_null_terminated_text(np.asarray(array[0], dtype=np.uint8)).strip()


def _add_circle_geometry(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    stimulus = root["analysis/stimulus_runs/stimulus_1"]
    calibration = stimulus.require_group("calibration")
    geometry = calibration.require_group("arena_geometry")
    geometry.attrs.update(
        {
            "coordinate_frame": "arena_relative_canvas_px",
            "coordinate_origin": "top_left_of_active_arena",
            "experimental_area_shape": "CIRCLE",
            "experimental_area_center_x_px": 10.0,
            "experimental_area_center_y_px": 10.0,
            "experimental_area_radius_px": 15.0,
            "experimental_area_radius_mm": 7.5,
        }
    )


def _write_sources(zarr_path: Path) -> str:
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)
    _add_circle_geometry(zarr_path)
    endpoint = build_cra_primary_endpoint_result(zarr_path, chaser_distance_run="chaser_distance_1")
    return write_cra_primary_endpoint_component(zarr_path, endpoint, overwrite=True, write_png=False)


def test_hysteresis_visits_counts_entries_and_closes_on_invalid() -> None:
    distance = np.asarray([6.0, 4.0, 4.5, 6.5, 4.0, np.nan, 4.0, 7.0], dtype=np.float32)
    valid = np.isfinite(distance)

    count, rate, median_dwell, total_dwell = compute_hysteresis_visits(
        distance,
        valid,
        fps=2.0,
        r_in_mm=5.0,
        r_out_mm=6.0,
    )

    assert count == 3
    assert math.isclose(rate, 45.0)
    assert math.isclose(median_dwell, 0.5)
    assert math.isclose(total_dwell, 2.0)


def test_available_annulus_area_uses_circular_arena_mask() -> None:
    geometry = ArenaGeometry(
        status="circle",
        source="test",
        width_px=20.0,
        height_px=20.0,
        shape="circle",
        center_x_px=10.0,
        center_y_px=10.0,
        radius_px=10.0,
    )
    circular_area = _available_annulus_area_mm2(
        geometry=geometry,
        object_x_px=18.0,
        object_y_px=10.0,
        pixels_per_mm=1.0,
        bin_edges_mm=np.asarray([0.0, 5.0], dtype=np.float32),
        grid_step_mm=0.25,
    )
    rectangle_area = _available_annulus_area_mm2(
        geometry=ArenaGeometry(
            status="rectangular_approximation",
            source="test",
            width_px=20.0,
            height_px=20.0,
            shape="rectangle",
            center_x_px=None,
            center_y_px=None,
            radius_px=None,
        ),
        object_x_px=18.0,
        object_y_px=10.0,
        pixels_per_mm=1.0,
        bin_edges_mm=np.asarray([0.0, 5.0], dtype=np.float32),
        grid_step_mm=0.25,
    )

    assert circular_area.shape == (1,)
    assert rectangle_area.shape == (1,)
    assert 0.0 < float(circular_area[0]) < float(rectangle_area[0])


def test_build_and_write_cra_near_field_component_from_existing_cra_stack(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    cra_component_path = _write_sources(zarr_path)

    result = build_cra_near_field_result(
        zarr_path,
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

    assert result.source_cra_primary_endpoint_path == cra_component_path
    assert result.geometry_status == "circle"
    assert result.arena_geometry_source == "analysis/stimulus_runs/stimulus_1/calibration/arena_geometry"
    # This fixture carries no analysis_metadata.dish_mask, so the resolver correctly falls back
    # to the projector's nominal experimental_area circle -- and must say so. A silent fallback
    # is the bug that inverted thigmotaxis on the real recording.
    assert result.qc_warnings == ("arena_geometry_fallback_to_nominal:circle",)
    assert result.geometry_status == "circle"
    assert result.near_zone_occupancy_fraction.shape == (2, 2)
    assert result.approach_percentile_mm.shape == (2, 2, 2)
    assert result.approach_percentile_cdf_fraction.shape == (2, 2, 2)
    assert result.object_distance_to_wall_mm.shape == (2, 2)
    assert result.radial_count.shape == (2, 2, 3)
    assert result.radial_count_wall_excluded.shape == (2, 2, 3)
    assert result.control_reference_labels == ("dish_center",)
    assert result.control_reference_cdf_fraction.shape == (2, 1, 2)
    assert result.cdf_fraction.shape == (2, 2, 2)
    assert result.mean_speed_mm_s.shape == (2,)
    assert result.immobile_fraction.shape == (2,)
    assert result.summary["approach_percentile_cdf_max_abs_error"] is not None
    assert math.isclose(result.summary["nearzone_occ_pre_agg"], 1.0)
    assert math.isclose(result.summary["nearzone_occ_post_agg"], 0.0)
    assert math.isclose(result.summary["nearzone_occ_delta_agg"], -1.0)
    assert math.isclose(result.summary["nearzone_occ_specificity"], -1.0)
    assert result.summary["delta_occ_agg"] == -1.0

    component_path = write_cra_near_field_component(zarr_path, result, overwrite=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    assert run[COMPONENT_PARENT_NAME].attrs["latest_complete"] == DEFAULT_COMPONENT_NAME
    component = root[component_path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["status"] == "computed"
    assert component.attrs["geometry_status"] == "circle"
    assert component.attrs["arena_shape"] == "circle"
    assert component.attrs["source_refs"]["source_cra_primary_endpoint_path"] == cra_component_path
    assert _decode_first(component["objects"]["behavior_class_label_bytes"]) == "aggressive"
    assert component["per_object_phase"]["near_zone_occupancy_fraction"][:].tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert "approach_percentile_cdf_fraction" in component["per_object_phase"]
    assert "object_distance_to_wall_mm" in component["per_object_phase"]
    assert "radial_density_wall_excluded_per_mm2" in component["radial_density"]
    assert "control_references" in component
    assert component["summary"]["nearzone_occ_delta_agg"][:].tolist() == [-1.0]
    assert component["thigmotaxis"].attrs["arena_shape"] == "circle"
    assert "mean_speed_mm_s" in component["thigmotaxis"]

    visualizations = component["visualizations"]
    for artifact_name in (
        RADIAL_DENSITY_PNG_ARTIFACT_NAME,
        DISTANCE_CDF_PNG_ARTIFACT_NAME,
        SUMMARY_PNG_ARTIFACT_NAME,
    ):
        png = visualizations[artifact_name]
        assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
        assert np.asarray(png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG")

    spec = visualizations[INTERACTIVE_ARTIFACT_NAME]
    assert spec.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec.attrs["renderer"] == "palette-goodcopbadcop-cra-near-field-v1"
