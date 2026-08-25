from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.provider_chaser_position_suite import (
    CircularArena,
    PositionSuiteEpoch,
)
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    ChaserRadialNearFieldInput,
    ChaserRadialNearFieldSuccessorError,
    _exact_time_visits,
    prepare_chaser_radial_near_field_successor,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.plot_chaser_radial_near_field_successor import render


SHA = "a" * 64


def _inputs(provider_id: str = "detection_bbox_centroid.v1") -> ChaserRadialNearFieldInput:
    n_frames = 8
    distance = np.asarray(
        [
            [7.0, 10.0],
            [4.0, 10.0],
            [4.0, 10.0],
            [7.0, 10.0],
            [7.0, 10.0],
            [4.0, 10.0],
            [7.0, 10.0],
            [7.0, 10.0],
        ],
        dtype=np.float64,
    )
    fish = np.broadcast_to(np.asarray([50.0, 50.0]), (n_frames, 2)).copy()
    chaser = np.empty((n_frames, 2, 2), dtype=np.float64)
    chaser[..., 1] = 50.0
    chaser[..., 0] = 50.0 + distance
    return ChaserRadialNearFieldInput(
        recording_id="recording",
        source_relative_frame_run_path="analysis/chaser_relative_frame_runs/exact",
        source_relative_frame_manifest_sha256=SHA,
        source_semantic_selection_run_path=(
            "analysis/protocol_semantic_chaser_selection_runs/exact"
        ),
        source_semantic_selection_manifest_sha256="b" * 64,
        fish_position_authority={
            "provider_id": provider_id,
            "provider_digest": "c" * 64,
            "coordinate_authority_id": "/coordinate@pixel_frame",
        },
        timing_authority={
            "timestamp_field": "timestamp_ns_session",
            "timing_authority_id": "analysis/source/selected_timestamp_ns_session",
            "timing_digest": "d" * 64,
        },
        arena_geometry_authority={
            "selection_record_sha256": "e" * 64,
            "physical_authority_sha256": "f" * 64,
            "pixel_frame_record_ref": "/coordinate@pixel_frame",
            "pixel_frame_record_sha256": "1" * 64,
        },
        n_frames=n_frames,
        n_chasers=2,
        acquisition_frame_id=np.arange(n_frames, dtype=np.int64),
        timestamp_ns_session=(
            np.asarray([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64) * 1_000_000_000
        ),
        timestamp_valid=np.ones(n_frames, dtype=bool),
        fish_xy_px=fish,
        fish_valid=np.ones(n_frames, dtype=bool),
        chaser_xy_px=chaser,
        chaser_valid=np.ones((n_frames, 2), dtype=bool),
        distance_px=distance,
        distance_px_valid=np.ones((n_frames, 2), dtype=bool),
        distance_mm=distance,
        distance_mm_valid=np.ones((n_frames, 2), dtype=bool),
        selection_member=np.ones(n_frames, dtype=bool),
        chaser_occurrence_member=np.ones((n_frames, 2), dtype=bool),
        chaser_role_codes=np.broadcast_to(
            np.asarray([1, 2], dtype=np.uint8), (n_frames, 2)
        ).copy(),
        chaser_role_valid=np.ones((n_frames, 2), dtype=bool),
        chaser_identity_codes=np.broadcast_to(
            np.asarray([1, 2], dtype=np.uint16), (n_frames, 2)
        ).copy(),
        role_registry={"1": "aggressive", "2": "inert"},
        chaser_registry={"1": "red", "2": "green"},
        epochs=(
            PositionSuiteEpoch(
                analysis_role="training",
                window_id=2,
                source_label="training",
                start_frame=0,
                end_frame=8,
                source_interval_sha256="2" * 64,
            ),
        ),
        arena=CircularArena(
            center_x_px=50.0,
            center_y_px=50.0,
            radius_px=100.0,
            boundary_role="physical_inner_rim",
            observed_feature="dish_inner_rim_water_side_edge",
        ),
        mm_per_pixel=1.0,
    )


@pytest.mark.parametrize(
    "provider_id",
    ["detection_bbox_centroid.v1", "keypoint_anatomical_triad_mean.v1"],
)
def test_position_providers_are_first_class_peers(provider_id: str) -> None:
    prepared = prepare_chaser_radial_near_field_successor(_inputs(provider_id))

    assert prepared.manifest["position_provider"] == {
        "provider_id": provider_id,
        "provider_digest": "c" * 64,
        "status": "first_class_explicit_authority",
        "provider_selection": "none",
    }
    assert prepared.manifest["source_distance_surface"]["unit"] == "mm"
    assert prepared.arrays["metric_distance_p50_mm"].tolist() == [7.0, 10.0]


def test_temporal_metrics_use_exact_irregular_session_intervals() -> None:
    prepared = prepare_chaser_radial_near_field_successor(_inputs())

    assert prepared.arrays["metric_near_zone_dwell_s"][0] == pytest.approx(3.0)
    assert prepared.arrays["metric_near_zone_valid_tracked_duration_s"][0] == pytest.approx(8.0)
    assert prepared.arrays["metric_near_zone_entry_count"][0] == 2
    assert prepared.arrays["metric_near_zone_entry_rate_per_min_valid_time"][0] == pytest.approx(15.0)
    assert prepared.arrays["metric_near_zone_complete_visit_total_dwell_s"][0] == pytest.approx(3.0)
    assert prepared.manifest["temporal_metric_timebase"]["physical_presentation_verified"] is False


def test_nonadjacent_frames_censor_visits_instead_of_bridging() -> None:
    result = _exact_time_visits(
        frame_id=np.asarray([0, 1, 2, 4, 5], dtype=np.int64),
        timestamp_ns=np.asarray([0, 1, 2, 4, 5], dtype=np.int64) * 1_000_000_000,
        timestamp_valid=np.ones(5, dtype=bool),
        distance_mm=np.asarray([7.0, 4.0, 4.0, 4.0, 7.0]),
        distance_valid=np.ones(5, dtype=bool),
        near_zone_mm=5.0,
        enter_mm=5.0,
        exit_mm=6.0,
    )

    assert result.entry_count == 1
    assert result.invalid_gap_count == 1
    assert result.invalid_gap_censor_event_count >= 1
    assert result.complete_visit_total_dwell_s == 0.0
    assert result.near_dwell_s == pytest.approx(2.0)


def test_missing_exact_timestamp_authority_fails_closed() -> None:
    inputs = _inputs()
    with pytest.raises(
        ChaserRadialNearFieldSuccessorError,
        match="timestamp_ns_session",
    ):
        prepare_chaser_radial_near_field_successor(
            replace(
                inputs,
                timing_authority={
                    **inputs.timing_authority,
                    "timestamp_field": None,
                },
            )
        )


def test_coordinate_authority_mismatch_fails_closed() -> None:
    inputs = _inputs()
    with pytest.raises(
        ChaserRadialNearFieldSuccessorError,
        match="coordinate authorities",
    ):
        prepare_chaser_radial_near_field_successor(
            replace(
                inputs,
                arena_geometry_authority={
                    **inputs.arena_geometry_authority,
                    "pixel_frame_record_ref": "/different@pixel_frame",
                },
            )
        )


def test_successor_publication_deep_audits_and_rehydrates(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording"
    prepared = prepare_chaser_radial_near_field_successor(_inputs())
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name="radial-v1", prepared=prepared
    )
    receipt = publish_composable_chaser_successor_run(
        plan, scratch_root=tmp_path / "scratch"
    )
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name="radial-v1",
        deep_audit=True,
    )
    reused = handle.prepared_successor()

    assert receipt["selector_eligible"] is False
    assert handle.run_path == "analysis/chaser_radial_near_field_runs/radial-v1"
    assert reused.payload_digest == prepared.payload_digest
    np.testing.assert_array_equal(
        reused.array("metric_distance_p50_mm"),
        prepared.array("metric_distance_p50_mm"),
    )
    plot = render(archive, run_name="radial-v1", output_stem=tmp_path / "radial")
    assert Path(plot["files"]["png"]["path"]).is_file()
    assert Path(plot["files"]["pdf"]["path"]).is_file()
    assert plot["source"]["manifest_sha256"] == handle.manifest_sha256
    assert plot["schema_version"] == 2
    assert plot["plot_parameters"]["scientific_coordinates"][
        "near_zone_radius_mm"
    ] == 5.0
    assert plot["plot_parameters"]["rendering"]["png_dpi"] == 180
