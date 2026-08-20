from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.utils.materialize_provider_spatial_canary import (
    CANARY_DISPOSITION,
    ProviderSpatialCanaryError,
    _occupancy_authorities,
    _source_coordinate_authority_id,
    load_task,
    planned_run_names,
)


def _task(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    recording_id = "recording-canary"
    archive = tmp_path / f"{recording_id}_analysis.zarr"
    archive.mkdir()
    payload: dict[str, object] = {
        "schema_id": "palette.provider_spatial_canary_task",
        "schema_version": 1,
        "disposition": CANARY_DISPOSITION,
        "campaign_id": "provider-spatial-v1",
        "recording_id": recording_id,
        "analysis_zarr": str(archive),
        "arena_id": 2,
        "subject_id": "subject-1",
        "epoch_source": {"run_name": "epoch-v2", "manifest_sha256": "a" * 64},
        "geometry_source": {
            "selection_run_name": "geometry-selection-v1",
            "selection_record_sha256": "b" * 64,
            "physical_authority_sha256": "c" * 64,
        },
        "recording_timing_authority_sha256": "d" * 64,
        "providers": {
            "detection": {
                "position_run_path": "analysis/subject_position_runs/observation/detection-v1",
                "position_manifest_sha256": "e" * 64,
                "tracking_run_path": None,
                "tracking_manifest_sha256": None,
            },
            "keypoint": {
                "position_run_path": "analysis/subject_position_runs/observation/keypoint-v1",
                "position_manifest_sha256": "f" * 64,
                "tracking_run_path": "tracking_runs/keypoint-track-v1",
                "tracking_manifest_sha256": "1" * 64,
            },
        },
        "selections": {
            "black_before": {"window_ids": [0], "role": "black_before"},
            "chaser": {"window_ids": [1], "role": "chaser"},
            "black_after": {"window_ids": [2], "role": "black_after"},
        },
        "grid": {"policy_id": "goodbatbadbat-arena-mm-v1", "bin_width_mm": 1.0},
        "contrasts": [
            {"baseline": "black_before", "treatment": "chaser"},
            {"baseline": "black_before", "treatment": "black_after"},
        ],
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    return task_path, payload


def test_task_freezes_two_providers_three_roles_and_exact_outputs(tmp_path: Path) -> None:
    task_path, _ = _task(tmp_path)
    task = load_task(task_path)
    outputs = planned_run_names(task)
    assert set(task["providers"]) == {"detection", "keypoint"}
    assert set(task["selections"]) == {"black_before", "chaser", "black_after"}
    assert outputs["tracking"]["keypoint"] == "keypoint-track-v1"
    assert outputs["tracking"]["detection"].startswith("tracking_detection_")
    assert len(outputs["contrasts"]["detection"]) == 2
    assert len(outputs["contrasts"]["keypoint"]) == 2


def test_task_rejects_inferred_or_pooled_first_canary_role(tmp_path: Path) -> None:
    task_path, payload = _task(tmp_path)
    payload["selections"]["black_before"] = {  # type: ignore[index]
        "window_ids": [0, 2],
        "role": "baseline",
    }
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProviderSpatialCanaryError, match="exactly one"):
        load_task(task_path)


def test_task_rejects_tracking_digest_without_exact_path(tmp_path: Path) -> None:
    task_path, payload = _task(tmp_path)
    payload["providers"]["detection"]["tracking_manifest_sha256"] = "9" * 64  # type: ignore[index]
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProviderSpatialCanaryError, match="requires a tracking path"):
        load_task(task_path)


def test_task_rejects_duplicate_directed_contrast(tmp_path: Path) -> None:
    task_path, payload = _task(tmp_path)
    payload["contrasts"] = [
        {"baseline": "black_before", "treatment": "chaser"},
        {"baseline": "black_before", "treatment": "chaser"},
    ]
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProviderSpatialCanaryError, match="duplicate directed"):
        load_task(task_path)


def test_source_coordinate_authority_unwraps_validated_descriptor_record() -> None:
    class Position:
        coordinate_record = {
            "coordinate_descriptor": {
                "frame_record": {"record_sha256": "a" * 64}
            }
        }

    assert _source_coordinate_authority_id(Position()) == "a" * 64


def test_occupancy_authority_keeps_grid_edges_array_backed() -> None:
    class GridPolicy:
        policy_id = "grid-v1"
        geometry = SimpleNamespace(
            geometry_id="geometry-v1",
            record_ref="geometry-ref",
            record_sha256="a" * 64,
            boundary_role="physical_inner_rim",
            observed_feature="dish_inner_rim_water_side_edge",
        )
        selection = SimpleNamespace(record_sha256="b" * 64)
        scale = SimpleNamespace(record_sha256="c" * 64)

        @staticmethod
        def source_binding_authority_record():
            return {
                "schema_id": "palette.arena_mm_grid_source_binding",
                "schema_version": 1,
                "grid_policy_id": "grid-v1",
                "bin_width_mm": 1.0,
                "extent_rule_id": "symmetric_outward_v1",
                "record_sha256": "d" * 64,
            }

    transform = SimpleNamespace(
        target_coordinate_authority_id="arena-mm-v1",
        source_coordinate_authority_id="camera-v1",
        sha256="e" * 64,
        matrix=np.eye(3, dtype=np.float64),
        source_camera_extent_px=(0.0, 100.0, 0.0, 100.0),
        grid_extent_mm=(-41.0, 41.0, -41.0, 41.0),
    )
    position = SimpleNamespace(
        manifest_sha256="f" * 64,
        estimator_record={"estimator_id": "detection_bbox_centroid.v1"},
    )
    evidence = SimpleNamespace(
        source_id="source-v1",
        sha256="1" * 64,
        record={
            "authorities": {"timeline_authority_id": "timeline-v1"},
            "schema_id": "palette.provider_spatial_track_source",
            "schema_version": 2,
        },
    )
    timing = SimpleNamespace(
        sha256="2" * 64,
        record={"nominal_fps": 100.0},
    )
    result = SimpleNamespace(
        config_digest="3" * 64,
        edge_policy_id="left_closed_right_open_final_inclusive_v1",
        timing_policy_id="valid_in_grid_sample_count_divided_by_fps_v1",
        fps_hz=100.0,
        x_edges=np.arange(-41.0, 42.0, dtype=np.float64),
        y_edges=np.arange(-41.0, 42.0, dtype=np.float64),
    )

    authority = _occupancy_authorities(
        {"recording_id": "recording-v1", "subject_id": "subject-v1"},
        provider="detection",
        position=position,
        evidence=evidence,
        timing=timing,
        grid_policy=GridPolicy(),
        transform=transform,
        result=result,
    )["fixed_grid_policy"]

    assert "x_edges" not in authority
    assert "y_edges" not in authority
    assert "x_edges_float64" not in authority["grid_policy"]
    assert "y_edges_float64" not in authority["grid_policy"]
    assert authority["recording_id"] == "recording-v1"
    assert authority["edge_count_xy"] == {"x": 83, "y": 83}
    assert authority["edge_array_paths"]["x"] == "grid/x_edges"
