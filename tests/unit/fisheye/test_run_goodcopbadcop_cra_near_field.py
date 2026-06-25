from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fisheye.utils import run_goodcopbadcop_cra_near_field as mod


def test_run_for_targets_forwards_near_field_parameters_and_writes(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    zarr_path = tmp_path / "recording_GoodCopBadCop_analysis.zarr"
    zarr_path.mkdir()

    def fake_build(path: Path, **kwargs):
        calls["build_path"] = path
        calls["build_kwargs"] = kwargs
        return SimpleNamespace(
            chaser_distance_run_name="chaser_distance_1",
            source_cra_primary_endpoint_component="object_relative_pre_post_v1",
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

    monkeypatch.setattr(mod, "build_cra_near_field_result", fake_build)
    monkeypatch.setattr(mod, "write_cra_near_field_component", fake_write)

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
        cra_primary_endpoint_component="latest",
        component_name="object_relative_near_field_v1",
        r_zone_mm=4.0,
        r_in_mm=4.0,
        r_out_mm=5.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=3.0,
        immobility_speed_threshold_mm_s=1.5,
        apply=True,
        overwrite=True,
        no_png=True,
        no_interactive_spec=True,
    )

    assert calls["build_path"] == zarr_path
    assert calls["build_kwargs"] == {
        "chaser_distance_run": "latest",
        "cra_primary_endpoint_component": "latest",
        "component_name": "object_relative_near_field_v1",
        "r_zone_mm": 4.0,
        "r_in_mm": 4.0,
        "r_out_mm": 5.0,
        "percentile_values": (5.0, 10.0),
        "radial_bin_edges_mm": (0.0, 2.0, 4.0),
        "cdf_thresholds_mm": (2.0, 4.0),
        "perimeter_band_mm": 3.0,
        "immobility_speed_threshold_mm_s": 1.5,
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
            "cra_primary_endpoint_component": "object_relative_pre_post_v1",
            "component_name": "object_relative_near_field_v1",
            "cra_near_field_path": "analysis/chaser_distance_runs/chaser_distance_1/cra_near_field/object_relative_near_field_v1",
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
