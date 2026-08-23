from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.materialize_provider_chaser_position_suite_cohort_canary import (
    ProviderChaserPositionSuiteCohortError,
    _aggregate_reports,
    load_cohort_task,
    resolve_cohort_task,
)


def _write_attrs(path: Path, attributes: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attributes,
            }
        ),
        encoding="utf-8",
    )


def _recording_fixture(root: Path, recording_id: str, camera: str) -> dict[str, object]:
    arena = "arena_" + recording_id.split("_arena_", 1)[1].split("_", 1)[0]
    archive = root / recording_id / f"{recording_id}_analysis.zarr"
    provider_run = f"sealed_detection_{recording_id[:10]}_{arena}"
    provider_manifest = {
        "recording_id": recording_id,
        "run_name": provider_run,
    }
    _write_attrs(
        archive / "analysis" / "provider_chaser_distance_runs" / provider_run,
        {
            "provider_chaser_distance_manifest": provider_manifest,
            "provider_chaser_distance_manifest_sha256": canonical_json_sha256(
                provider_manifest
            ),
        },
    )
    geometry_run = f"arena_geometry_selection_{camera}_{recording_id[:10]}"
    _write_attrs(
        archive / "analysis" / "arena_geometry_selection",
        {"latest": geometry_run, "latest_complete": geometry_run},
    )
    selection_record = {
        "selected_candidate": {
            "arena_binding": {"arena_id": arena, "camera_serial": camera}
        }
    }
    _write_attrs(
        archive / "analysis" / "arena_geometry_selection" / geometry_run,
        {
            "selection_id": geometry_run,
            "selection_record": selection_record,
            "selection_record_sha256": canonical_json_sha256(selection_record),
        },
    )
    physical = {"camera_id": camera, "mm_per_pixel": 0.02}
    _write_attrs(
        archive / "analysis" / "calibration" / "coordinate_frames",
        {
            "source_camera_physical_authority": physical,
            "source_camera_physical_authority_sha256": canonical_json_sha256(physical),
        },
    )
    return {
        "recording_id": recording_id,
        "analysis_zarr": str(archive),
        "providers": {"detection_bbox_centroid": {"sealed_run_name": provider_run}},
    }


def _cohort_source(tmp_path: Path) -> Path:
    entries = []
    for arena_number in range(1, 5):
        camera = str(2010092 + arena_number)
        for day in ("2026-08-10", "2026-08-12"):
            recording_id = f"{day}T17-20-55Z_arena_{arena_number}_goodbatbadbat"
            entries.append(
                _recording_fixture(tmp_path / "recordings", recording_id, camera)
            )
    source = tmp_path / "cohort_inputs.json"
    source.write_text(
        json.dumps(
            {
                "schema_id": "palette.provider_chaser_distance.cohort_inputs",
                "schema_version": 1,
                "recording_count": len(entries),
                "plan_digest": "a" * 64,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return source


def test_plan_freezes_exact_provider_geometry_and_physical_authorities(
    tmp_path: Path,
) -> None:
    source = _cohort_source(tmp_path)
    task = resolve_cohort_task(source)

    assert len(task["entries"]) == 8
    assert task["selection"]["selected_recording_count"] == 8
    assert len(task["task_sha256"]) == 64
    assert all("latest" not in row["geometry_selection_run"] for row in task["entries"])
    assert all(len(row["provider_manifest_sha256"]) == 64 for row in task["entries"])
    assert task["aggregation_policy"]["frame_pooling_across_recordings"] is False
    assert load_cohort_task(task) == task


def test_frozen_task_rejects_changed_source_cohort(tmp_path: Path) -> None:
    source = _cohort_source(tmp_path)
    task = resolve_cohort_task(source)
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ProviderChaserPositionSuiteCohortError,
        match="absent or has changed",
    ):
        load_cohort_task(task)


def test_planning_fails_when_geometry_selectors_disagree(tmp_path: Path) -> None:
    source = _cohort_source(tmp_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    archive = Path(document["entries"][0]["analysis_zarr"])
    parent = archive / "analysis" / "arena_geometry_selection" / "zarr.json"
    metadata = json.loads(parent.read_text(encoding="utf-8"))
    metadata["attributes"]["latest"] = "different_selection"
    parent.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ProviderChaserPositionSuiteCohortError,
        match="Geometry selectors disagree",
    ):
        resolve_cohort_task(source)


def _report(recording_id: str, value: float, frame_count: int) -> dict[str, object]:
    metrics = []
    contrasts = []
    radial = []
    for window_id, epoch in enumerate(("pre", "training", "post")):
        for role in ("aggressive", "inert"):
            role_offset = 0.0 if role == "aggressive" else 1.0
            metrics.append(
                {
                    "analysis_role": epoch,
                    "epoch_window_id": window_id,
                    "behavior_role": role,
                    "epoch_provider_frame_coverage_fraction": 0.9,
                    "valid_distance_fraction": 0.8,
                    "distance_p50_mm": value + role_offset,
                    "same_quadrant_fraction_valid": 0.5,
                    "near_zone_fraction_valid": 0.1,
                    "near_zone_entry_rate_per_min_valid_time": 2.0,
                    "fish_arena_radius_mean_mm": 20.0,
                    "fish_wall_distance_mean_mm": 10.0,
                }
            )
            radial.append(
                {
                    "analysis_role": epoch,
                    "epoch_window_id": window_id,
                    "behavior_role": role,
                    "bin_start_mm": 0.0,
                    "bin_end_mm": 2.0,
                    "observed_fraction": 0.1 + role_offset,
                    "expected_fraction_geometric": 0.2,
                    "selection_index_geometric": value,
                }
            )
        contrasts.append(
            {
                "analysis_role": epoch,
                "epoch_window_id": window_id,
                "metric": "distance_p50_mm",
                "treatment_minus_baseline": -1.0,
            }
        )
    return {
        "recording_id": recording_id,
        "source_bindings": {
            "provider_chaser_distance": {
                "manifest_sha256": "a" * 64,
                "source_position_provider": {
                    "coordinate_authority_id": (
                        "/analysis/coordinate_frames/source_camera/2010093/"
                        "continuous@pixel_frame_authority"
                    )
                },
            },
            "arena_geometry_and_scale": {
                "selection": {"sha256": "b" * 64},
                "source_camera_physical_authority": {"sha256": "c" * 64},
            },
        },
        "suite": {
            "frame_count": frame_count,
            "mm_per_pixel": 0.02,
            "arena": {"radius_mm": 40.0},
            "per_epoch_chaser_metrics": metrics,
            "role_contrasts": contrasts,
            "radial_occupancy": radial,
        },
    }


def test_aggregation_uses_one_value_per_recording_not_frame_pooling() -> None:
    reports = [
        _report(
            "2026-08-10T17-20-55Z_arena_1_goodbatbadbat",
            value=10.0,
            frame_count=10,
        ),
        _report(
            "2026-08-12T17-20-55Z_arena_1_goodbatbadbat",
            value=20.0,
            frame_count=1_000_000,
        ),
    ]
    outputs = _aggregate_reports(reports)
    recording_rows, _, _, _, distributions, radial_summary, _ = outputs

    assert len(recording_rows) == 2
    distance = [
        row
        for row in distributions
        if row["analysis_role"] == "pre"
        and row["behavior_role"] == "aggressive"
        and row["metric"] == "distance_p50_mm"
    ]
    assert len(distance) == 1
    assert distance[0]["recording_count"] == 2
    assert distance[0]["p50"] == pytest.approx(15.0)
    assert radial_summary[0]["selection_index_geometric_recording_count"] == 2
