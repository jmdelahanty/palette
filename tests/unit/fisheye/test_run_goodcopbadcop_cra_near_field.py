from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.utils import run_goodcopbadcop_cra_near_field as mod
from fisheye.utils import run_chaser_near_field_occupancy as generic_mod


def test_run_for_targets_forwards_near_field_parameters_and_writes(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    zarr_path = tmp_path / "recording_GoodCopBadCop_analysis.zarr"
    zarr_path.mkdir()
    dependency_handle = {"record_sha256": "a" * 64}

    def fake_build(path: Path, **kwargs):
        calls["build_path"] = path
        calls["build_kwargs"] = kwargs
        return SimpleNamespace(
            chaser_distance_run_name="chaser_distance_1",
            source_quadrant_occupancy_component="object_relative_pre_post_v1",
            source_quadrant_occupancy_manifest_sha256="b" * 64,
            geometry_status="circle",
            summary={
                "approach_p05_delta_agg": -1.25,
                "nearzone_occ_delta_agg": 0.02,
                "nearzone_entry_rate_delta_agg": 0.5,
            },
            endpoint_status="computed",
        )

    def fake_write(path: Path, result, **kwargs):
        calls["write_path"] = path
        calls["write_result"] = result
        calls["write_kwargs"] = kwargs
        return "analysis/chaser_distance_runs/chaser_distance_1/cra_near_field/object_relative_near_field_v1"

    monkeypatch.setattr(generic_mod, "build_chaser_near_field_occupancy_result", fake_build)
    monkeypatch.setattr(generic_mod, "write_chaser_near_field_occupancy_component", fake_write)

    rows = mod.run_for_targets(
        [
            {
                "recording_id": "recording_GoodCopBadCop",
                "zarr_path": str(zarr_path),
                "coverage_percent": 99.0,
                "detect_run": "detect_1",
                "refined_run": "refined_1",
            }
        ],
        chaser_distance_run="latest",
        quadrant_occupancy_component="latest",
        quadrant_occupancy_dependency_handle=dependency_handle,
        legacy_quadrant_occupancy_component_compatibility=False,
        component_name="object_relative_near_field_v1",
        r_zone_mm=4.0,
        r_in_mm=4.0,
        r_out_mm=5.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=3.0,
        immobility_speed_threshold_mm_s=1.5,
        immobility_signal_mode="verified_track_motion",
        apply=True,
        overwrite=True,
        no_png=True,
        no_interactive_spec=True,
    )

    assert calls["build_path"] == zarr_path
    assert calls["build_kwargs"] == {
        "chaser_distance_run": "latest",
        "quadrant_occupancy_component": "latest",
        "quadrant_occupancy_dependency_handle": dependency_handle,
        "legacy_quadrant_occupancy_component_compatibility": False,
        "component_name": "object_relative_near_field_v1",
        "r_zone_mm": 4.0,
        "r_in_mm": 4.0,
        "r_out_mm": 5.0,
        "percentile_values": (5.0, 10.0),
        "radial_bin_edges_mm": (0.0, 2.0, 4.0),
        "cdf_thresholds_mm": (2.0, 4.0),
        "perimeter_band_mm": 3.0,
        "immobility_speed_threshold_mm_s": 1.5,
        "immobility_signal_mode": "verified_track_motion",
    }
    assert calls["write_path"] == zarr_path
    assert calls["write_kwargs"] == {
        "overwrite": True,
        "write_png": False,
        "write_interactive_spec": False,
    }
    assert rows == [
        {
            "recording_id": "recording_GoodCopBadCop",
            "zarr_path": str(zarr_path),
            "detect_coverage_percent": 99.0,
            "detect_run": "detect_1",
            "refined_run": "refined_1",
            "chaser_distance_run": "chaser_distance_1",
            "quadrant_occupancy_component": "object_relative_pre_post_v1",
            "quadrant_occupancy_manifest_sha256": "b" * 64,
            "component_name": "object_relative_near_field_v1",
            "chaser_near_field_occupancy_path": "analysis/chaser_distance_runs/chaser_distance_1/cra_near_field/object_relative_near_field_v1",
            "geometry_status": "circle",
            "status": "computed",
            "error": None,
            "summary": {
                "approach_p05_delta_agg": -1.25,
                "nearzone_occ_delta_agg": 0.02,
                "nearzone_entry_rate_delta_agg": 0.5,
            },
        }
    ]


def test_exact_named_component_is_converted_to_dependency_handle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    component = object()
    root = {"analysis/chaser_distance_runs/base/chaser_quadrant_occupancy/exact": component}
    snapshot = SimpleNamespace(run_path="analysis/chaser_distance_runs/base")
    expected_handle = {"record_sha256": "c" * 64}
    calls: list[tuple[object, object, str]] = []
    monkeypatch.setattr(generic_mod, "open_zarr_root", lambda *_a, **_k: root)
    monkeypatch.setattr(
        generic_mod,
        "load_chaser_distance_run",
        lambda candidate_root, *, run_name: (
            snapshot
            if candidate_root is root and run_name == "base"
            else pytest.fail("wrong base resolution")
        ),
    )

    def fake_build_handle(candidate, *, snapshot, relative_path):
        calls.append((candidate, snapshot, relative_path))
        return expected_handle

    monkeypatch.setattr(
        generic_mod,
        "build_chaser_component_handle",
        fake_build_handle,
    )

    actual = generic_mod._build_explicit_quadrant_dependency_handle(
        tmp_path / "analysis.zarr",
        chaser_distance_run="base",
        quadrant_occupancy_component="exact",
    )

    assert actual is expected_handle
    assert calls == [(component, snapshot, "chaser_quadrant_occupancy/exact")]


def test_batch_runner_refuses_implicit_latest_before_builder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    zarr_path.mkdir()
    monkeypatch.setattr(
        generic_mod,
        "build_chaser_near_field_occupancy_result",
        lambda *_a, **_k: pytest.fail("implicit latest reached scientific builder"),
    )

    rows = generic_mod.run_for_targets(
        [{"recording_id": "recording", "zarr_path": str(zarr_path)}],
        chaser_distance_run="base",
        quadrant_occupancy_component="latest",
        component_name="near",
        r_zone_mm=4.0,
        r_in_mm=4.0,
        r_out_mm=5.0,
        percentile_values=(5.0,),
        radial_bin_edges_mm=None,
        cdf_thresholds_mm=(2.0,),
        perimeter_band_mm=3.0,
        immobility_speed_threshold_mm_s=1.5,
        immobility_signal_mode="verified_track_motion",
        apply=False,
        overwrite=False,
        no_png=True,
        no_interactive_spec=True,
    )

    assert rows[0]["status"] == "failed"
    assert "requires an exact" in rows[0]["error"]


def test_parse_float_list_accepts_none_string_and_sequence() -> None:
    assert mod._parse_float_list(None) is None
    assert mod._parse_float_list("1, 2.5,3") == [1.0, 2.5, 3.0]
    assert mod._parse_float_list((4, 5.5)) == [4.0, 5.5]


def test_filesystem_targets_discovers_recording_zarrs(tmp_path: Path) -> None:
    included = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop" / "zarr" / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr"
    excluded = tmp_path / "2026-06-14T21-12-08Z_arena_1_Feeding" / "zarr" / "2026-06-14T21-12-08Z_arena_1_Feeding_analysis.zarr"
    included.mkdir(parents=True)
    excluded.mkdir(parents=True)

    targets = mod._filesystem_targets(
        [tmp_path],
        recording_like="%GoodCopBadCop%",
        limit=None,
        recursive=False,
    )

    assert targets == [
        {
            "recording_id": "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
            "zarr_path": str(included.resolve()),
            "coverage_percent": None,
            "detect_run": None,
            "model_name": None,
            "refined_run": None,
        }
    ]
