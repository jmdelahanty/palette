from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis.provider_chaser_position_suite import (
    CircularArena,
    PositionSuiteEpoch,
)
from fisheye.analysis_workflows.chaser_near_field_visit_successor import (
    ChaserNearFieldVisitInput,
    ChaserNearFieldVisitSuccessorError,
    RADIAL_PARITY_ARRAY_NAMES,
    prepare_chaser_near_field_visit_successor,
)
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    ChaserRadialNearFieldInput,
    prepare_chaser_radial_near_field_successor,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.near_field_visit_state_machine import (
    LEFT_CENSOR_INVALID_GAP,
    LEFT_CENSOR_PHASE_START,
    RIGHT_CENSOR_INVALID_GAP,
    RIGHT_CENSOR_PHASE_END,
    segment_exact_time_near_field_visits,
)
from fisheye.shared.zarr_io import open_zarr_root

SHA = "a" * 64


def _radial_inputs() -> ChaserRadialNearFieldInput:
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
            "provider_id": "detection_bbox_centroid.v1",
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


def _visit_inputs() -> ChaserNearFieldVisitInput:
    source = _radial_inputs()
    radial = prepare_chaser_radial_near_field_successor(source)
    return ChaserNearFieldVisitInput(
        recording_id=source.recording_id,
        source_relative_frame_run_path=source.source_relative_frame_run_path,
        source_relative_frame_manifest_sha256=(
            source.source_relative_frame_manifest_sha256
        ),
        source_semantic_selection_run_path=(source.source_semantic_selection_run_path),
        source_semantic_selection_manifest_sha256=(
            source.source_semantic_selection_manifest_sha256
        ),
        source_radial_near_field_run_path=(
            "analysis/chaser_radial_near_field_runs/radial"
        ),
        source_radial_near_field_manifest_sha256="9" * 64,
        radial_near_field_manifest=radial.manifest,
        radial_metric_arrays={
            name: radial.array(name) for name in RADIAL_PARITY_ARRAY_NAMES
        },
        fish_position_authority=source.fish_position_authority,
        timing_authority=source.timing_authority,
        n_frames=source.n_frames,
        n_chasers=source.n_chasers,
        acquisition_frame_id=source.acquisition_frame_id,
        track_sample_id=np.arange(source.n_frames, dtype=np.int64) + 100,
        timestamp_ns_session=source.timestamp_ns_session,
        timestamp_valid=source.timestamp_valid,
        fish_source_row_id=np.arange(source.n_frames, dtype=np.int64) + 200,
        fish_source_row_valid=np.ones(source.n_frames, dtype=bool),
        fish_xy_px=source.fish_xy_px,
        fish_valid=source.fish_valid,
        chaser_source_row_id=np.arange(
            source.n_frames * source.n_chasers, dtype=np.int64
        ).reshape(source.n_frames, source.n_chasers),
        chaser_source_row_valid=np.ones(
            (source.n_frames, source.n_chasers), dtype=bool
        ),
        chaser_xy_px=source.chaser_xy_px,
        chaser_valid=source.chaser_valid,
        relative_vector_mm=source.chaser_xy_px - source.fish_xy_px[:, None, :],
        distance_mm=source.distance_mm,
        distance_valid=source.distance_mm_valid,
        selection_member=source.selection_member,
        chaser_occurrence_member=source.chaser_occurrence_member,
        chaser_role_code=source.chaser_role_codes,
        chaser_role_valid=source.chaser_role_valid,
        chaser_identity_code=source.chaser_identity_codes,
        role_registry=source.role_registry,
        chaser_registry=source.chaser_registry,
        epochs=source.epochs,
        arena_center_xy_px=np.asarray([50.0, 50.0], dtype=np.float64),
        arena_radius_px=100.0,
        mm_per_pixel=1.0,
    )


def test_state_machine_retains_boundary_and_gap_censors() -> None:
    boundary = segment_exact_time_near_field_visits(
        frame_id=np.arange(3, dtype=np.int64),
        timestamp_ns=np.arange(3, dtype=np.int64) * 1_000_000_000,
        timestamp_valid=np.ones(3, dtype=bool),
        distance_mm=np.asarray([4.0, 4.0, 7.0]),
        distance_valid=np.ones(3, dtype=bool),
        near_zone_mm=5.0,
        enter_mm=5.0,
        exit_mm=6.0,
    )
    assert len(boundary.visits) == 1
    assert boundary.visits[0].entry_observed is False
    assert boundary.visits[0].exit_observed is True
    assert boundary.visits[0].left_censor_reason_code == LEFT_CENSOR_PHASE_START
    assert boundary.boundary_censor_event_count == 1

    gap = segment_exact_time_near_field_visits(
        frame_id=np.arange(6, dtype=np.int64),
        timestamp_ns=np.arange(6, dtype=np.int64) * 1_000_000_000,
        timestamp_valid=np.ones(6, dtype=bool),
        distance_mm=np.asarray([7.0, 4.0, 4.0, np.nan, 4.0, 7.0]),
        distance_valid=np.asarray([True, True, True, False, True, True]),
        near_zone_mm=5.0,
        enter_mm=5.0,
        exit_mm=6.0,
    )
    assert len(gap.visits) == 2
    assert gap.visits[0].right_censor_reason_code == RIGHT_CENSOR_INVALID_GAP
    assert gap.visits[1].left_censor_reason_code == LEFT_CENSOR_INVALID_GAP
    assert gap.entry_count == 1
    assert gap.invalid_gap_count == 1
    assert gap.invalid_gap_censor_event_count == 2


def test_state_machine_threshold_equality_and_phase_end_are_not_crossings() -> None:
    result = segment_exact_time_near_field_visits(
        frame_id=np.arange(5, dtype=np.int64),
        timestamp_ns=np.arange(5, dtype=np.int64) * 1_000_000_000,
        timestamp_valid=np.ones(5, dtype=bool),
        distance_mm=np.asarray([7.0, 5.0, 4.0, 6.0, 6.0]),
        distance_valid=np.ones(5, dtype=bool),
        near_zone_mm=5.0,
        enter_mm=5.0,
        exit_mm=6.0,
    )

    assert len(result.visits) == 1
    visit = result.visits[0]
    assert visit.first_sample_index == 2
    assert visit.last_inside_index == 4
    assert visit.exit_observed is False
    assert visit.right_censor_reason_code == RIGHT_CENSOR_PHASE_END
    assert result.complete_visit_total_dwell_s == 0.0


def test_successor_persists_every_short_visit_and_exact_samples() -> None:
    prepared = prepare_chaser_near_field_visit_successor(_visit_inputs())

    assert prepared.n_visits == 2
    assert prepared.n_samples == 3
    assert prepared.n_summary_rows == 2
    assert prepared.array("visit_sample_offset").tolist() == [0, 2]
    assert prepared.array("visit_sample_count").tolist() == [2, 1]
    assert prepared.array("visit_quality_code").tolist() == [1, 1]
    assert prepared.array("visit_complete").tolist() == [True, True]
    assert prepared.array("visit_entry_acquisition_frame_id").tolist() == [1, 5]
    assert prepared.array("visit_exit_acquisition_frame_id").tolist() == [3, 6]
    assert prepared.array("sample_acquisition_frame_id").tolist() == [1, 2, 5]
    assert prepared.array("sample_fish_source_row_id").tolist() == [201, 202, 205]
    assert prepared.array("sample_chaser_source_row_id").tolist() == [2, 4, 10]
    np.testing.assert_allclose(prepared.array("sample_canonical_x_mm"), [4.0, 4.0, 4.0])
    np.testing.assert_allclose(
        prepared.array("sample_canonical_y_mm"), [0.0, 0.0, 0.0], atol=1e-12
    )
    assert prepared.array("summary_observed_entry_visit_count").tolist() == [2, 0]
    assert prepared.array("summary_complete_visit_count").tolist() == [2, 0]
    assert prepared.array("summary_complete_visit_total_dwell_s")[0] == pytest.approx(
        3.0
    )
    assert prepared.array("summary_short_visit_count").tolist() == [2, 0]
    assert prepared.array("visit_key_sha256_bytes").shape == (2, 32)
    assert not np.array_equal(
        prepared.array("visit_key_sha256_bytes")[0],
        prepared.array("visit_key_sha256_bytes")[1],
    )
    assert prepared.array("visit_row_id").flags.writeable is False
    assert prepared.manifest["radial_aggregate_parity"]["status"] == "exact"


def test_radial_aggregate_mismatch_fails_closed() -> None:
    inputs = _visit_inputs()
    arrays = {
        name: np.array(value, copy=True)
        for name, value in inputs.radial_metric_arrays.items()
    }
    arrays["metric_near_zone_entry_count"][0] += 1
    with pytest.raises(
        ChaserNearFieldVisitSuccessorError,
        match="differs from radial aggregate",
    ):
        prepare_chaser_near_field_visit_successor(
            replace(inputs, radial_metric_arrays=arrays)
        )


def test_relative_vector_direction_must_match_sealed_positions() -> None:
    inputs = _visit_inputs()

    with pytest.raises(
        ChaserNearFieldVisitSuccessorError,
        match="disagree with the sealed fish/chaser positions",
    ):
        prepare_chaser_near_field_visit_successor(
            replace(inputs, relative_vector_mm=-inputs.relative_vector_mm)
        )


def test_short_visit_threshold_changes_quality_not_membership() -> None:
    inputs = _visit_inputs()
    permissive = prepare_chaser_near_field_visit_successor(
        replace(inputs, minimum_quality_sample_count=1)
    )
    strict = prepare_chaser_near_field_visit_successor(
        replace(inputs, minimum_quality_sample_count=10)
    )

    assert permissive.n_visits == strict.n_visits == 2
    assert permissive.n_samples == strict.n_samples == 3
    assert permissive.array("visit_quality_code").tolist() == [0, 0]
    assert strict.array("visit_quality_code").tolist() == [1, 1]


def test_visit_successor_publication_round_trip(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording"
    prepared = prepare_chaser_near_field_visit_successor(_visit_inputs())
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name="visits-v1",
        prepared=prepared,
    )
    receipt = publish_composable_chaser_successor_run(
        plan,
        scratch_root=tmp_path / "scratch",
    )
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_near_field_visits",
        run_name="visits-v1",
        deep_audit=True,
    )
    reused = handle.prepared_successor()

    assert receipt["selector_eligible"] is False
    assert handle.run_path == "analysis/chaser_near_field_visits_runs/visits-v1"
    assert reused.payload_digest == prepared.payload_digest
    np.testing.assert_array_equal(
        reused.array("sample_acquisition_frame_id"),
        prepared.array("sample_acquisition_frame_id"),
    )
