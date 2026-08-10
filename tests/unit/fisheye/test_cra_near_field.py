from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis.chaser_near_field_occupancy as near_field_module
import fisheye.analysis.cra_near_field as cra_compatibility_module
from fisheye.analysis.chaser_component_publication import (
    ChaserComponentContract,
    build_chaser_component_handle,
    component_record_sha256,
    persist_chaser_component_manifest,
)
from fisheye.analysis.chaser_component_writer import _STAGING_CAPABILITY
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.chaser_near_field_occupancy import (
    COMPONENT_PARENT_NAME,
    DISTANCE_CDF_PNG_ARTIFACT_NAME,
    INTERACTIVE_ARTIFACT_NAME,
    INTERACTIVE_RENDERER,
    LEGACY_VISIT_POLICY,
    RADIAL_DENSITY_PNG_ARTIFACT_NAME,
    SCHEMA_ID,
    SUMMARY_PNG_ARTIFACT_NAME,
    VALID_TRACKED_VISIT_POLICY,
    ArenaGeometry,
    _available_annulus_area_mm2,
    _resolve_quadrant_occupancy_component,
    build_chaser_near_field_occupancy_result as build_cra_near_field_result,
    compute_hysteresis_visits,
    write_chaser_near_field_occupancy_component as write_cra_near_field_component,
)
from fisheye.analysis.chaser_profiles import default_goodcopbadcop_source_profile_path
from fisheye.analysis.chaser_quadrant_occupancy import (
    DEFAULT_COMPONENT_NAME as DEFAULT_QUADRANT_COMPONENT_NAME,
    METHOD as QUADRANT_METHOD,
    METHOD_VERSION as QUADRANT_METHOD_VERSION,
    SCHEMA_ID as QUADRANT_SCHEMA_ID,
    SCHEMA_VERSION as QUADRANT_SCHEMA_VERSION,
    build_chaser_quadrant_occupancy_result as build_cra_primary_endpoint_result,
    write_chaser_quadrant_occupancy_component as write_cra_primary_endpoint_component,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID
from tests.unit.fisheye.test_cra_primary_endpoint import _make_archive, _make_chaser_result


pytestmark = pytest.mark.usefixtures("logical_chaser_distance_reader")


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
    write_chaser_distance_run(
        zarr_path,
        _make_chaser_result(zarr_path),
        overwrite=True,
        legacy_compatibility=True,
    )
    _add_circle_geometry(zarr_path)
    endpoint = build_cra_primary_endpoint_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        protocol_profile=default_goodcopbadcop_source_profile_path(),
    )
    component_path = write_cra_primary_endpoint_component.__wrapped__(
        zarr_path,
        endpoint,
        overwrite=True,
        write_png=False,
        write_interactive_spec=False,
        _chaser_component_staging_capability=_STAGING_CAPABILITY,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    relative_path = str(component_path).removeprefix(f"{snapshot.run_path}/")
    persist_chaser_component_manifest(
        root[str(component_path)],
        snapshot=snapshot,
        relative_path=relative_path,
        contract=ChaserComponentContract(
            component_family="chaser_quadrant_occupancy",
            component_name=DEFAULT_QUADRANT_COMPONENT_NAME,
            semantic_schema_id=QUADRANT_SCHEMA_ID,
            semantic_schema_version=QUADRANT_SCHEMA_VERSION,
            method_id=QUADRANT_METHOD,
            method_version=QUADRANT_METHOD_VERSION,
            parameters={"fixture": "exact_quadrant_dependency"},
            source_authorities={"fixture": "sealed"},
        ),
    )
    root[str(component_path)].attrs["palette_run_completion_status"] = "complete"
    root[str(component_path)].attrs["stage_selector_eligible"] = True
    return str(component_path)


def _quadrant_handle(zarr_path: Path, component_path: str) -> dict[str, object]:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    relative_path = component_path.removeprefix(f"{snapshot.run_path}/")
    return build_chaser_component_handle(
        root[component_path],
        snapshot=snapshot,
        relative_path=relative_path,
    )


def _resolve_quadrant_fixture(
    zarr_path: Path,
    *,
    handle: dict[str, object] | None,
    legacy_compatibility: bool,
):
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    return _resolve_quadrant_occupancy_component(
        root,
        snapshot=snapshot,
        run_group=root[snapshot.run_path],
        chaser_distance_run_path=snapshot.run_path,
        component_name="latest",
        dependency_handle=handle,
        legacy_compatibility=legacy_compatibility,
    )


def test_quadrant_latest_discovery_requires_explicit_compatibility(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    component_path = _write_sources(zarr_path)

    with pytest.raises(ValueError, match="explicit self-digested"):
        _resolve_quadrant_fixture(
            zarr_path,
            handle=None,
            legacy_compatibility=False,
        )

    _component, name, path, manifest_sha256 = _resolve_quadrant_fixture(
        zarr_path,
        handle=None,
        legacy_compatibility=True,
    )
    assert name == DEFAULT_QUADRANT_COMPONENT_NAME
    assert path == component_path
    assert manifest_sha256 is None


@pytest.mark.parametrize(
    "field",
    ["component_path", "component_manifest_sha256", "record_sha256"],
)
def test_invalid_explicit_quadrant_handle_never_falls_back_to_latest(
    tmp_path: Path,
    field: str,
) -> None:
    zarr_path = _make_archive(tmp_path)
    component_path = _write_sources(zarr_path)
    invalid = copy.deepcopy(_quadrant_handle(zarr_path, component_path))
    if field != "record_sha256":
        invalid[field] = (
            f"{invalid[field]}-wrong"
            if field == "component_path"
            else "0" * 64
        )
        body = {key: value for key, value in invalid.items() if key != "record_sha256"}
        invalid["record_sha256"] = component_record_sha256(body)
    else:
        invalid[field] = "0" * 64

    with pytest.raises(ValueError):
        _resolve_quadrant_fixture(
            zarr_path,
            handle=invalid,
            legacy_compatibility=True,
        )


def test_quadrant_handle_rejects_different_base_snapshot(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    component_path = _write_sources(zarr_path)
    source_handle = copy.deepcopy(_quadrant_handle(zarr_path, component_path))
    source_handle["base_publication_seal_sha256"] = "f" * 64
    body = {
        key: value
        for key, value in source_handle.items()
        if key != "record_sha256"
    }
    source_handle["record_sha256"] = component_record_sha256(body)

    with pytest.raises(ValueError, match="different base authority"):
        _resolve_quadrant_fixture(
            zarr_path,
            handle=source_handle,
            legacy_compatibility=True,
        )


def test_quadrant_handle_rejects_wrong_component_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        near_field_module,
        "open_explicit_chaser_component_group",
        lambda *_args, **_kwargs: SimpleNamespace(
            component_family="wrong_family",
            component_name="sealed_quadrant",
            component_path=(
                "analysis/chaser_distance_runs/run/wrong_family/sealed_quadrant"
            ),
            manifest_sha256="a" * 64,
            group={},
        ),
    )

    with pytest.raises(ValueError, match="different component family"):
        _resolve_quadrant_occupancy_component(
            {},
            snapshot=SimpleNamespace(run_path="analysis/chaser_distance_runs/run"),
            run_group={},
            chaser_distance_run_path="analysis/chaser_distance_runs/run",
            component_name="latest",
            dependency_handle={"explicit": True},
            legacy_compatibility=True,
        )


def test_historical_cra_alias_enables_only_explicit_legacy_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        cra_compatibility_module,
        "build_chaser_near_field_occupancy_result",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "result",
    )

    result = cra_compatibility_module.build_cra_near_field_result(
        "archive.zarr",
        cra_primary_endpoint_component="object_relative_pre_post_v1",
    )

    assert result == "result"
    assert calls == [
        (
            ("archive.zarr",),
            {
                "quadrant_occupancy_component": "latest",
                "legacy_quadrant_occupancy_component_compatibility": True,
            },
        )
    ]


def test_historical_cra_alias_cannot_weaken_exact_handle_path() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        cra_compatibility_module.build_cra_near_field_result(
            "archive.zarr",
            cra_primary_endpoint_component="legacy",
            quadrant_occupancy_dependency_handle={"record_sha256": "a" * 64},
        )


def test_hysteresis_visits_censors_dropout_without_synthetic_reentry() -> None:
    distance = np.asarray([7.0, 4.0, 4.5, 6.5, 4.0, np.nan, 4.0, 7.0], dtype=np.float32)
    valid = np.isfinite(distance)

    result = compute_hysteresis_visits(
        distance,
        valid,
        fps=2.0,
        r_in_mm=5.0,
        r_out_mm=6.0,
    )

    assert result.policy_version == VALID_TRACKED_VISIT_POLICY
    assert result.entry_count == 2
    assert result.valid_sample_count == 7
    assert math.isclose(result.valid_tracked_duration_s, 3.5)
    assert math.isclose(result.rate_denominator_duration_s, 3.5)
    assert math.isclose(result.entry_rate_per_min, 2.0 / (3.5 / 60.0))
    assert math.isclose(result.complete_visit_median_dwell_s, 1.0)
    assert math.isclose(result.complete_visit_total_dwell_s, 1.0)
    assert result.invalid_gap_count == 1
    assert result.invalid_gap_censor_event_count == 2


def test_hysteresis_visits_records_boundary_censoring() -> None:
    result = compute_hysteresis_visits(
        np.asarray([4.0, 4.5, 7.0, 4.0, 4.5], dtype=np.float32),
        np.ones(5, dtype=bool),
        fps=1.0,
        r_in_mm=5.0,
        r_out_mm=6.0,
    )

    assert result.entry_count == 1
    assert math.isclose(result.entry_rate_per_min, 12.0)
    assert result.censor_event_count == 2
    assert result.boundary_censor_event_count == 2
    assert math.isnan(result.complete_visit_median_dwell_s)
    assert result.complete_visit_total_dwell_s == 0.0


def test_hysteresis_visits_all_invalid_has_no_rate_denominator() -> None:
    result = compute_hysteresis_visits(
        np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
        np.zeros(3, dtype=bool),
        fps=10.0,
        r_in_mm=5.0,
        r_out_mm=6.0,
    )

    assert result.entry_count == 0
    assert result.valid_sample_count == 0
    assert result.valid_tracked_duration_s == 0.0
    assert result.rate_denominator_duration_s == 0.0
    assert math.isnan(result.entry_rate_per_min)
    assert result.invalid_gap_count == 1


def test_hysteresis_visits_rate_uses_exact_valid_tracking_time() -> None:
    result = compute_hysteresis_visits(
        np.asarray([7.0, 4.0, 7.0, np.nan, 7.0, 4.0, 7.0], dtype=np.float32),
        np.asarray([True, True, True, False, True, True, True]),
        fps=2.0,
        r_in_mm=5.0,
        r_out_mm=6.0,
    )

    assert result.entry_count == 2
    assert result.valid_sample_count == 6
    assert math.isclose(result.valid_tracked_duration_s, 3.0)
    assert math.isclose(result.rate_denominator_duration_s, 3.0)
    assert math.isclose(result.entry_rate_per_min, 40.0)
    assert math.isclose(result.complete_visit_median_dwell_s, 0.5)
    assert math.isclose(result.complete_visit_total_dwell_s, 1.0)


def test_hysteresis_visits_legacy_gap_split_requires_explicit_version() -> None:
    result = compute_hysteresis_visits(
        np.asarray([6.0, 4.0, 4.5, 6.5, 4.0, np.nan, 4.0, 7.0]),
        np.asarray([True, True, True, True, True, False, True, True]),
        fps=2.0,
        r_in_mm=5.0,
        r_out_mm=6.0,
        policy_version=LEGACY_VISIT_POLICY,
    )

    assert result.policy_version == LEGACY_VISIT_POLICY
    assert result.entry_count == 3
    assert math.isclose(result.entry_rate_per_min, 45.0)
    assert math.isclose(result.valid_tracked_duration_s, 3.5)
    assert math.isclose(result.rate_denominator_duration_s, 4.0)
    assert result.rate_denominator_semantics == (
        "all_effective_phase_frames_divided_by_fps"
    )


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
        chaser_x_px=18.0,
        chaser_y_px=10.0,
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
        chaser_x_px=18.0,
        chaser_y_px=10.0,
        pixels_per_mm=1.0,
        bin_edges_mm=np.asarray([0.0, 5.0], dtype=np.float32),
        grid_step_mm=0.25,
    )

    assert circular_area.shape == (1,)
    assert rectangle_area.shape == (1,)
    assert 0.0 < float(circular_area[0]) < float(rectangle_area[0])


def test_build_and_write_cra_near_field_component_from_existing_cra_stack(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    cra_component_path = _write_sources(zarr_path)
    quadrant_handle = _quadrant_handle(zarr_path, cra_component_path)

    result = build_cra_near_field_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        quadrant_occupancy_component=DEFAULT_QUADRANT_COMPONENT_NAME,
        quadrant_occupancy_dependency_handle=quadrant_handle,
        r_zone_mm=2.0,
        r_in_mm=2.0,
        r_out_mm=3.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0, 8.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=2.0,
        immobility_signal_mode="raw_centroid_explicit",
    )

    assert result.source_quadrant_occupancy_path == cra_component_path
    assert (
        result.source_quadrant_occupancy_manifest_sha256
        == quadrant_handle["component_manifest_sha256"]
    )
    assert result.geometry_status == "circle"
    assert result.arena_geometry_source == "analysis/stimulus_runs/stimulus_1/calibration/arena_geometry"
    # This fixture carries no analysis_metadata.dish_mask, so the resolver correctly falls back
    # to the projector's nominal experimental_area circle -- and must say so. A silent fallback
    # is the bug that inverted thigmotaxis on the real recording.
    assert result.qc_warnings == (
        "immobility_signal_explicit_raw_centroid",
        "arena_geometry_fallback_to_nominal:circle",
    )
    assert result.speed_source == "raw_centroid_explicit"
    assert result.geometry_status == "circle"
    assert result.near_zone_occupancy_fraction.shape == (2, 2)
    assert result.approach_percentile_mm.shape == (2, 2, 2)
    assert result.approach_percentile_cdf_fraction.shape == (2, 2, 2)
    assert result.chaser_distance_to_wall_mm.shape == (2, 2)
    assert result.radial_count.shape == (2, 2, 3)
    assert result.radial_count_wall_excluded.shape == (2, 2, 3)
    assert result.control_reference_labels == ("dish_center",)
    assert result.control_reference_cdf_fraction.shape == (2, 1, 2)
    assert result.cdf_fraction.shape == (2, 2, 2)
    assert result.mean_speed_mm_s.shape == (2,)
    assert result.immobile_fraction.shape == (2,)
    assert result.summary["approach_percentile_cdf_max_abs_error"] is not None
    aggressive = result.summary["per_chaser"][0]
    assert math.isclose(
        aggressive["phase_values"][0]["near_zone_occupancy_fraction"],
        1.0,
    )
    assert math.isclose(
        aggressive["phase_values"][1]["near_zone_occupancy_fraction"],
        0.0,
    )

    component_path = write_cra_near_field_component(zarr_path, result, overwrite=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    assert "latest" not in run[COMPONENT_PARENT_NAME].attrs
    assert "latest_complete" not in run[COMPONENT_PARENT_NAME].attrs
    component = root[component_path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["status"] == "computed"
    assert component.attrs["geometry_status"] == "circle"
    assert component.attrs["arena_shape"] == "circle"
    assert (
        component.attrs["source_refs"]["source_quadrant_occupancy_path"]
        == cra_component_path
    )
    assert (
        component.attrs["source_refs"][
            "source_quadrant_occupancy_manifest_sha256"
        ]
        == quadrant_handle["component_manifest_sha256"]
    )
    assert (
        _decode_first(component["chasers"]["behavior_class_label_bytes"])
        == "aggressive"
    )
    assert component["per_chaser_phase"]["near_zone_occupancy_fraction"][
        :
    ].tolist() == [[1.0, 0.0], [0.0, 0.0]]
    per_chaser = component["per_chaser_phase"]
    assert "near_zone_entry_rate_numerator_count" in per_chaser
    assert "near_zone_valid_tracked_duration_s" in per_chaser
    assert "near_zone_entry_rate_denominator_duration_s" in per_chaser
    assert "near_zone_censor_event_count" in per_chaser
    assert per_chaser.attrs["visit_policy_version"] == VALID_TRACKED_VISIT_POLICY
    assert per_chaser.attrs["near_zone_entry_rate_denominator"].startswith(
        "near_zone_entry_rate_denominator_duration_s"
    )
    assert "approach_percentile_cdf_fraction" in component["per_chaser_phase"]
    assert "chaser_distance_to_wall_mm" in component["per_chaser_phase"]
    assert "radial_density_wall_excluded_per_mm2" in component["radial_density"]
    assert "control_references" in component
    assert "per_chaser_json_bytes" in component["summary"]
    assert component["thigmotaxis"].attrs["arena_shape"] == "circle"
    assert "mean_speed_mm_s" in component["thigmotaxis"]
    assert _decode_first(component["thigmotaxis"]["speed_source_bytes"]) == (
        "raw_centroid_explicit"
    )

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
    assert spec.attrs["renderer"] == INTERACTIVE_RENDERER


def test_verified_track_failure_does_not_fall_back_to_raw_centroid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "verified_track_failure"
    case_path.mkdir()
    zarr_path = _make_archive(case_path)
    quadrant_path = _write_sources(zarr_path)
    quadrant_handle = _quadrant_handle(zarr_path, quadrant_path)
    monkeypatch.setattr(
        near_field_module,
        "load_verified_smoothed_frame_speed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("track motion manifest mismatch")
        ),
    )

    with pytest.raises(ValueError, match="track motion manifest mismatch"):
        build_cra_near_field_result(
            zarr_path,
            chaser_distance_run="chaser_distance_1",
            quadrant_occupancy_dependency_handle=quadrant_handle,
        )


def test_verified_speed_authority_controls_published_phase_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "verified_track_speed"
    case_path.mkdir()
    zarr_path = _make_archive(case_path)
    quadrant_path = _write_sources(zarr_path)
    quadrant_handle = _quadrant_handle(zarr_path, quadrant_path)
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    total_frames = int(
        root["analysis/chaser_distance_runs/chaser_distance_1/positions/fish_valid"].shape[0]
    )
    verified_values = np.zeros(total_frames, dtype=np.float64)
    verified_values[0] = np.nan
    authority = {
        "schema_id": "palette.track_motion_read_authority",
        "schema_version": 1,
        "run_ref": "analysis/track_kinematics_runs/offline/tk_verified",
        "track_ref": "analysis/track_kinematics_runs/offline/tk_verified/tracks/id_0",
    }
    monkeypatch.setattr(
        near_field_module,
        "load_verified_smoothed_frame_speed",
        lambda _root, frame_count: SimpleNamespace(
            values_mm_s=verified_values,
            source="track_motion.movement/speed/smoothed/mm",
            authority=authority,
        )
        if frame_count == total_frames
        else pytest.fail("wrong frame extent"),
    )

    result = build_cra_near_field_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        quadrant_occupancy_dependency_handle=quadrant_handle,
    )

    assert result.speed_source == "track_motion.movement/speed/smoothed/mm"
    assert result.source_track_motion_authority == authority
    assert np.nanmax(result.mean_speed_mm_s) == pytest.approx(0.0)
    assert np.nanmin(result.immobile_fraction) == pytest.approx(1.0)
    assert "immobility_signal_explicit_raw_centroid" not in result.qc_warnings
