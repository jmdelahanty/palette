from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceResult,
    ChaserDistanceWindow,
    write_chaser_distance_run,
)
from fisheye.analysis.cra_primary_endpoint import (
    COMPONENT_PARENT_NAME,
    DEFAULT_COMPONENT_NAME,
    INTERACTIVE_ARTIFACT_NAME,
    INTERACTIVE_RENDERER,
    OVERVIEW_PNG_ARTIFACT_NAME,
    QUADRANT_LABELS,
    SCHEMA_ID,
    build_cra_primary_endpoint_result,
    quadrant_code_for_xy,
    resolve_effective_phase_windows,
    resolve_object_roles_from_protocol_payload,
    write_cra_primary_endpoint_component,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    data = np.asarray(values)
    group.create_array(name, data=data, chunks=data.shape, overwrite=True)


def _make_protocol_json(*, position_transition_duration_s: float = 0.1) -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "parameters": {
                        "position_transition_duration_s": position_transition_duration_s,
                        "pre_period_duration_s": 0.3,
                        "training_period_duration_s": 0.3,
                        "post_period_duration_s": 0.3,
                        "pixels_per_mm": 2.0,
                        "chasers": [
                            {
                                "enable_chase": True,
                                "behavior_mode": 0,
                                "color_r": 1.0,
                                "color_g": 0.0,
                                "color_b": 0.0,
                                "color_a": 1.0,
                                "start_position_preset": "top_left",
                                "end_position_preset": "bottom_right",
                            },
                            {
                                "enable_chase": False,
                                "behavior_mode": 1,
                                "color_r": 0.0,
                                "color_g": 0.0,
                                "color_b": 1.0,
                                "color_a": 1.0,
                                "start_position_preset": "top_right",
                                "end_position_preset": "bottom_left",
                            },
                        ],
                    }
                }
            ]
        }
    )


def _make_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "goodcopbadcop_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "test_GoodCopBadCop"
    analysis = root.create_group("analysis")
    stimulus_parent = analysis.create_group("stimulus_runs")
    stimulus_parent.attrs["latest"] = "stimulus_1"
    stimulus_parent.attrs["latest_complete"] = "stimulus_1"
    stimulus = stimulus_parent.create_group("stimulus_1")
    stimulus.attrs["protocol_json"] = _make_protocol_json()
    coords = stimulus.create_group("stimulus_coordinates")
    arena = coords.create_group("arena_1")
    arena.attrs.update(
        {
            "texture_width_px": 20.0,
            "texture_height_px": 20.0,
            "texture_origin": "top_left",
        }
    )
    return zarr_path


def _make_chaser_result(zarr_path: Path) -> ChaserDistanceResult:
    n = 9
    chasers = np.asarray([0, 1], dtype=np.uint8)
    camera_frame_id = np.arange(n, dtype=np.int64)
    fish_xy = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 3.0],
            [6.0, 4.0],
            [7.0, 12.0],
            [8.0, 12.0],
            [9.0, 12.0],
        ],
        dtype=np.float32,
    )
    chaser_xy = np.zeros((n, 2, 2), dtype=np.float32)
    chaser_xy[:6, 0, :] = np.asarray([0.0, 0.0], dtype=np.float32)
    chaser_xy[:6, 1, :] = np.asarray([10.0, 0.0], dtype=np.float32)
    chaser_xy[6:, 0, :] = np.asarray([15.0, 15.0], dtype=np.float32)
    chaser_xy[6:, 1, :] = np.asarray([5.0, 15.0], dtype=np.float32)
    distance_px = np.linalg.norm(fish_xy[:, None, :] - chaser_xy, axis=2).astype(np.float32)
    distance_mm = (distance_px / 2.0).astype(np.float32)
    windows = (
        ChaserDistanceWindow(0, "pre_event", 0, 2, 0.0, 0.3, 0.3),
        ChaserDistanceWindow(1, "training_event", 3, 5, 0.3, 0.6, 0.3),
        ChaserDistanceWindow(2, "post_event", 6, 8, 0.6, 0.9, 0.3),
    )
    hist_counts = np.ones((3, 2, 3), dtype=np.int64)
    hist_density = hist_counts.astype(np.float32)
    hist_density /= np.maximum(hist_density.sum(axis=2, keepdims=True), 1)
    return ChaserDistanceResult(
        zarr_path=str(zarr_path),
        recording_id="test_GoodCopBadCop",
        run_name="chaser_distance_1",
        source_detection_path="refined_detect_runs/refined_1/instances",
        source_detection_kind="refined",
        source_stimulus_run="stimulus_1",
        source_stimulus_path="analysis/stimulus_runs/stimulus_1",
        source_stimulus_epoch_run="epochs_1",
        source_stimulus_epoch_path="analysis/stimulus_epoch_runs/epochs_1",
        fps=10.0,
        total_frames=n,
        pixels_per_mm_projector=2.0,
        coordinate_frame="arena_relative_canvas_px",
        coordinate_origin="top_left_of_active_arena",
        arena_origin_in_canvas_xy=(0.0, 0.0),
        chaser_indices=chasers,
        chaser_behavior_class_id=np.asarray([1, 3], dtype=np.int8),
        chaser_behavior_labels=("aggressive", "inert"),
        camera_frame_id=camera_frame_id,
        stimulus_frame_num=camera_frame_id,
        timestamp_ns=np.arange(n, dtype=np.int64),
        stimulus_epoch_window_id=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
        fish_centroid_img_xy=fish_xy,
        fish_centroid_arena_xy=fish_xy,
        chaser_arena_xy=chaser_xy,
        fish_valid=np.asarray([True, True, True, True, False, True, True, True, True], dtype=bool),
        chaser_valid=np.ones((n, 2), dtype=bool),
        distance_px=distance_px,
        distance_mm=distance_mm,
        nearest_chaser_index=np.argmin(distance_mm, axis=1).astype(np.int16),
        nearest_distance_mm=np.min(distance_mm, axis=1).astype(np.float32),
        windows=windows,
        epoch_valid_frame_count=np.asarray([[3, 3], [2, 2], [3, 3]], dtype=np.int64),
        epoch_mean_distance_mm=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        epoch_min_distance_mm=np.asarray([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float32),
        epoch_p05_distance_mm=np.asarray([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float32),
        epoch_p50_distance_mm=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        epoch_p95_distance_mm=np.asarray([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32),
        epoch_fraction_within_threshold=np.asarray([[0.2, 0.4], [0.6, 0.8], [0.7, 0.9]], dtype=np.float32),
        threshold_mm=20.0,
        distribution_bin_width_mm=2.0,
        histogram_bin_edges_mm=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
        histogram_bin_centers_mm=np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
        histogram_counts=hist_counts,
        histogram_density=hist_density,
    )


def _decode_first(array: zarr.Array) -> str:
    return decode_null_terminated_text(np.asarray(array[0], dtype=np.uint8)).strip()


def test_resolve_object_roles_from_protocol_payload_maps_protocol_metadata() -> None:
    roles = resolve_object_roles_from_protocol_payload(json.loads(_make_protocol_json()))

    assert [(role.object_index, role.object_role) for role in roles] == [(0, "aggressive"), (1, "inert")]
    assert [role.behavior_class_id for role in roles] == [1, 3]
    assert roles[0].raw_color_hex == "#ff0000"
    assert roles[1].raw_color_hex == "#0000ff"
    assert roles[0].start_position_preset == "top_left"
    assert roles[0].end_position_preset == "bottom_right"


def test_effective_phase_windows_trim_post_settle_and_exclude_training() -> None:
    windows = (
        ChaserDistanceWindow(0, "pre_event", 0, 2, 0.0, 0.3, 0.3),
        ChaserDistanceWindow(1, "training_event", 3, 5, 0.3, 0.6, 0.3),
        ChaserDistanceWindow(2, "post_event", 6, 8, 0.6, 0.9, 0.3),
    )

    phases = resolve_effective_phase_windows(windows, fps=10.0, post_settle_duration_s=0.1)

    assert [phase.phase_label for phase in phases] == ["pre_static", "post_static"]
    assert [(phase.effective_start_frame, phase.effective_end_frame) for phase in phases] == [(0, 2), (7, 8)]
    assert phases[1].settle_excluded_frame_count == 1


def test_quadrant_code_for_xy_uses_right_bottom_midline_ownership() -> None:
    assert QUADRANT_LABELS[quadrant_code_for_xy(1.0, 1.0, width_px=20.0, height_px=20.0)] == "top_left"
    assert QUADRANT_LABELS[quadrant_code_for_xy(10.0, 1.0, width_px=20.0, height_px=20.0)] == "top_right"
    assert QUADRANT_LABELS[quadrant_code_for_xy(1.0, 10.0, width_px=20.0, height_px=20.0)] == "bottom_left"
    assert QUADRANT_LABELS[quadrant_code_for_xy(10.0, 10.0, width_px=20.0, height_px=20.0)] == "bottom_right"
    assert quadrant_code_for_xy(-1.0, 1.0, width_px=20.0, height_px=20.0) == -1


def test_build_and_write_cra_primary_endpoint_component(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)

    result = build_cra_primary_endpoint_result(zarr_path, chaser_distance_run="chaser_distance_1")

    assert result.endpoint_status == "computed"
    assert result.qc_warnings == ()
    assert result.quadrant_width_px == 20.0
    assert result.quadrant_height_px == 20.0
    assert result.object_quadrant_code.tolist() == [[0, 1], [3, 2]]
    assert result.occupancy_fraction.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert math.isclose(result.summary["occ_pre_agg"], 1.0)
    assert math.isclose(result.summary["occ_post_agg"], 0.0)
    assert math.isclose(result.summary["delta_occ_agg"], -1.0)
    assert math.isclose(result.summary["occ_post_inert"], 1.0)
    assert result.summary["occ_post_benign"] == result.summary["occ_post_inert"]
    assert result.summary["pre_aggressive_quadrant"] == "top_left"
    assert result.summary["post_aggressive_quadrant"] == "bottom_right"
    assert result.phases[1].effective_start_frame == 7
    assert result.phases[1].settle_excluded_frame_count == 1
    assert math.isclose(result.summary["d_pre_agg"], float(np.median(result.median_distance_mm[0, 0:1])))

    component_path = write_cra_primary_endpoint_component(zarr_path, result, overwrite=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    assert _decode_first(run["chasers"]["behavior_class_label_bytes"]) == "aggressive"
    assert np.asarray(run["chasers"]["behavior_class_id"][:]).tolist() == [1, 3]
    assert run[COMPONENT_PARENT_NAME].attrs["latest_complete"] == DEFAULT_COMPONENT_NAME
    component = root[component_path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["status"] == "computed"
    assert component.attrs["summary"]["delta_occ_agg"] == -1.0
    assert component["object_phase"]["object_quadrant_code"][:].tolist() == [[0, 1], [3, 2]]
    assert component["phases"]["effective_start_frame"][:].tolist() == [0, 7]
    assert _decode_first(component["objects"]["behavior_class_label_bytes"]) == "aggressive"
    assert float(component["summary"]["delta_occ_agg"][0]) == -1.0

    component_visualizations = component["visualizations"]
    png = component_visualizations[OVERVIEW_PNG_ARTIFACT_NAME]
    assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert png.attrs["canonical_artifact"] is True
    assert np.asarray(png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG")
    spec_group = component_visualizations[INTERACTIVE_ARTIFACT_NAME]
    assert spec_group.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec_group.attrs["renderer"] == INTERACTIVE_RENDERER
    assert spec_group.attrs["canonical_artifact"] is True

    mirror_spec = run["visualizations"][INTERACTIVE_ARTIFACT_NAME]
    assert mirror_spec.attrs["canonical_artifact"] is False
    assert mirror_spec.attrs["component_path"] == component_path
    assert mirror_spec.attrs["canonical_artifact_path"] == f"{component_path}/visualizations/{INTERACTIVE_ARTIFACT_NAME}"


def test_cra_primary_endpoint_rejects_noncanonical_chaser_distance_frame(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["analysis/chaser_distance_runs/chaser_distance_1"].attrs["coordinate_frame"] = "source_image_px"

    with pytest.raises(ValueError, match="requires coordinate_frame='arena_relative_canvas_px'"):
        build_cra_primary_endpoint_result(zarr_path, chaser_distance_run="chaser_distance_1")


def test_cra_primary_endpoint_rejects_noncanonical_chaser_distance_origin(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["analysis/chaser_distance_runs/chaser_distance_1"].attrs["coordinate_origin"] = "camera_top_left"

    with pytest.raises(ValueError, match="requires coordinate_origin='top_left_of_active_arena'"):
        build_cra_primary_endpoint_result(zarr_path, chaser_distance_run="chaser_distance_1")


def test_cra_primary_endpoint_uses_single_nondefault_stimulus_arena_bounds(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    coords = root["analysis/stimulus_runs/stimulus_1/stimulus_coordinates"]
    del coords["arena_1"]
    arena = coords.create_group("arena_2")
    arena.attrs.update(
        {
            "texture_width_px": 20.0,
            "texture_height_px": 20.0,
            "texture_origin": "top_left",
        }
    )
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)

    result = build_cra_primary_endpoint_result(zarr_path, chaser_distance_run="chaser_distance_1")

    assert result.quadrant_bounds_source == "analysis/stimulus_runs/*/stimulus_coordinates/arena_2"
    assert result.object_quadrant_code.tolist() == [[0, 1], [3, 2]]


def test_cra_primary_endpoint_dropout_warning_is_report_only_by_default(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    chaser_result = _make_chaser_result(zarr_path)
    fish_valid = chaser_result.fish_valid.copy()
    fish_valid[7:9] = False
    write_chaser_distance_run(
        zarr_path,
        replace(chaser_result, fish_valid=fish_valid),
        overwrite=True,
    )

    result = build_cra_primary_endpoint_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        dropout_warning_fraction=0.2,
    )

    assert result.endpoint_status == "computed"
    assert result.dropout_exclusion_fraction is None
    assert result.tracking_dropout_fraction[1, 0] == 1.0
    assert any("post_static:aggressive:tracking_dropout_fraction>0.2" == item for item in result.qc_warnings)

    component_path = write_cra_primary_endpoint_component(zarr_path, result, overwrite=True)
    root = zarr.open_group(str(zarr_path), mode="r")
    component = root[component_path]
    assert component.attrs["status"] == "computed"
    assert component.attrs["qc_warnings"] == list(result.qc_warnings)
