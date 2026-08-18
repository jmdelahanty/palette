from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.provider_occupancy_v2 import (
    EDGE_POLICY_ID,
    EMPTY_FRACTION_POLICY_ID,
    OccupancyGrid,
    OccupancyTimingPolicy,
    ProviderOccupancySamples,
    TIMING_POLICY_ID,
    build_provider_occupancy_config_digest,
    calculate_provider_occupancy_v2,
    occupancy_samples_from_trajectory,
)
from fisheye.analysis.provider_spatial_trajectory import (
    ProviderTrackSamples,
    SelectedFrameMembership,
    SourceCameraToArenaMMTransform,
    TrajectoryAuthorityIdentities,
    prepare_provider_spatial_trajectory,
    selected_frame_membership_from_compiled_selection,
)
from fisheye.analysis_workflows.composable_stimulus_selection import (
    RoleMetadata,
    SelectionSpec,
    TimelineAuthority,
    compile_selection,
    member,
    stimulus_step_reference,
    union,
)


def _samples(
    x: list[float],
    y: list[float],
    *,
    selected: list[bool] | None = None,
    provider_present: list[bool] | None = None,
    provider_valid: list[bool] | None = None,
    transform_valid: list[bool] | None = None,
    occurrences: list[str] | None = None,
) -> ProviderOccupancySamples:
    n = len(x)
    return ProviderOccupancySamples(
        x_mm=np.asarray(x, dtype=np.float64),
        y_mm=np.asarray(y, dtype=np.float64),
        selected=np.ones(n, dtype=bool) if selected is None else selected,
        provider_present=np.ones(n, dtype=bool)
        if provider_present is None
        else provider_present,
        provider_valid=np.ones(n, dtype=bool)
        if provider_valid is None
        else provider_valid,
        transform_valid=np.ones(n, dtype=bool)
        if transform_valid is None
        else transform_valid,
        occurrence_ids=np.asarray(
            ["default"] * n if occurrences is None else occurrences,
            dtype="U",
        ),
    )


def test_boundary_edges_are_left_closed_and_final_outer_edge_inclusive() -> None:
    grid = OccupancyGrid(x_edges=[0.0, 1.0, 2.0], y_edges=[0.0, 1.0, 2.0])
    result = calculate_provider_occupancy_v2(
        _samples(
            [0.0, 1.0, 2.0, -0.001, 2.001],
            [0.0, 1.0, 2.0, 1.0, 1.0],
        ),
        grid,
        OccupancyTimingPolicy(fps_hz=10.0),
    )

    np.testing.assert_array_equal(
        result.pooled.counts,
        np.asarray([[1, 0], [0, 2]], dtype=np.int64),
    )
    assert result.pooled.valid_in_grid_sample_count == 3
    assert result.pooled.out_of_grid_count == 2
    assert result.x_edges.dtype == np.float64
    assert result.y_edges.dtype == np.float64
    np.testing.assert_allclose(
        result.pooled.occupancy_time_by_bin_s,
        result.pooled.counts.astype(np.float64) / 10.0,
    )
    np.testing.assert_allclose(
        result.pooled.occupancy_fraction,
        np.asarray([[1 / 3, 0.0], [0.0, 2 / 3]], dtype=np.float64),
    )


def test_empty_selection_has_coverage_zero_and_undefined_nan_fraction() -> None:
    result = calculate_provider_occupancy_v2(
        _samples(
            [0.5, 1.5],
            [0.5, 1.5],
            selected=[False, False],
            occurrences=["pre", "post"],
        ),
        OccupancyGrid(x_edges=[0.0, 1.0, 2.0], y_edges=[0.0, 1.0, 2.0]),
        OccupancyTimingPolicy(fps_hz=20.0),
    )

    assert result.per_occurrence == ()
    assert result.pooled.expected_selected_frames == 0
    assert result.pooled.valid_in_grid_sample_count == 0
    assert result.pooled.occupancy_time_s == 0.0
    assert np.all(result.pooled.counts == 0)
    assert np.all(np.isnan(result.pooled.occupancy_fraction))
    result.validate_conservation()
    assert EMPTY_FRACTION_POLICY_ID == "nan_when_no_valid_in_grid_samples_v1"


def test_occurrences_and_pooling_keep_separate_counts_and_time() -> None:
    result = calculate_provider_occupancy_v2(
        _samples(
            [0.25, 1.25, 0.25],
            [0.25, 0.25, 1.25],
            occurrences=["pre", "pre", "chaser"],
        ),
        OccupancyGrid(x_edges=[0.0, 1.0, 2.0], y_edges=[0.0, 1.0, 2.0]),
        OccupancyTimingPolicy(fps_hz=10.0),
    )

    assert [item.occurrence_id for item in result.per_occurrence] == ["pre", "chaser"]
    np.testing.assert_array_equal(
        result.per_occurrence[0].counts,
        np.asarray([[1, 1], [0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.per_occurrence[1].counts,
        np.asarray([[0, 0], [1, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.pooled.counts,
        result.per_occurrence[0].counts + result.per_occurrence[1].counts,
    )
    assert result.per_occurrence[0].occupancy_time_s == 0.2
    assert result.pooled.occupancy_time_s == 0.3
    assert result.fps_hz == 10.0


def test_invalid_states_are_excluded_but_reported_separately() -> None:
    result = calculate_provider_occupancy_v2(
        _samples(
            [0.5, 0.5, 0.5, np.nan, 3.0, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            provider_present=[True, False, True, True, True, True],
            provider_valid=[True, False, False, True, True, True],
            transform_valid=[True, False, False, True, True, False],
        ),
        OccupancyGrid(x_edges=[0.0, 1.0, 2.0], y_edges=[0.0, 1.0, 2.0]),
        OccupancyTimingPolicy(fps_hz=5.0),
    )

    summary = result.pooled
    assert summary.expected_selected_frames == 6
    assert summary.provider_present_count == 5
    assert summary.provider_valid_count == 4
    assert summary.provider_missing_count == 1
    assert summary.provider_invalid_count == 1
    assert summary.transform_invalid_count == 1
    assert summary.nonfinite_count == 1
    assert summary.out_of_grid_count == 1
    assert summary.valid_in_grid_sample_count == 1
    assert summary.occupancy_time_s == 0.2
    assert int(summary.counts.sum()) == 1


def test_conservation_validation_rejects_tampered_summary() -> None:
    result = calculate_provider_occupancy_v2(
        _samples([0.5], [0.5]),
        OccupancyGrid(x_edges=[0.0, 1.0], y_edges=[0.0, 1.0]),
        OccupancyTimingPolicy(fps_hz=10.0),
    )
    summary = result.pooled
    tampered = type(summary)(
        occurrence_id=summary.occurrence_id,
        counts=np.zeros_like(summary.counts),
        occupancy_fraction=summary.occupancy_fraction,
        expected_selected_frames=summary.expected_selected_frames,
        provider_present_count=summary.provider_present_count,
        provider_valid_count=summary.provider_valid_count,
        transform_invalid_count=summary.transform_invalid_count,
        nonfinite_count=summary.nonfinite_count,
        out_of_grid_count=summary.out_of_grid_count,
        valid_in_grid_sample_count=summary.valid_in_grid_sample_count,
        occupancy_time_s=summary.occupancy_time_s,
    )
    with pytest.raises(ValueError, match="count conservation"):
        tampered.validate_conservation()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_edges": [0.0], "y_edges": [0.0, 1.0]},
        {"x_edges": [0.0, 1.0, 1.0], "y_edges": [0.0, 1.0]},
        {"x_edges": [1.0, 0.0], "y_edges": [0.0, 1.0]},
        {"x_edges": [0.0, np.nan], "y_edges": [0.0, 1.0]},
    ],
)
def test_malformed_grids_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        OccupancyGrid(**kwargs)


def test_malformed_cardinality_and_state_dependencies_fail_closed() -> None:
    with pytest.raises(ValueError, match="equal cardinality"):
        ProviderOccupancySamples(
            x_mm=[0.0, 1.0],
            y_mm=[0.0],
            selected=[True, True],
            provider_present=[True, True],
            provider_valid=[True, True],
            transform_valid=[True, True],
            occurrence_ids=["a", "a"],
        )
    with pytest.raises(ValueError, match="provider_valid"):
        _samples(
            [0.5],
            [0.5],
            provider_present=[False],
            provider_valid=[True],
            transform_valid=[False],
        )
    with pytest.raises(ValueError, match="transform_valid"):
        _samples(
            [0.5],
            [0.5],
            provider_present=[True],
            provider_valid=[False],
            transform_valid=[True],
        )


def test_config_digest_and_explicit_policies_are_deterministic() -> None:
    grid = OccupancyGrid(
        x_edges=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        y_edges=np.asarray([0.0, 1.0], dtype=np.float64),
    )
    timing = OccupancyTimingPolicy(fps_hz=50.0)
    first = build_provider_occupancy_config_digest(grid, timing)
    second = build_provider_occupancy_config_digest(
        OccupancyGrid([0.0, 1.0, 2.0], [0.0, 1.0]),
        OccupancyTimingPolicy(50.0),
    )
    assert first == second
    assert len(first) == 64
    assert grid.edge_policy_id == EDGE_POLICY_ID
    assert timing.timing_policy_id == TIMING_POLICY_ID


def test_expected_frame_denominator_includes_selected_frames_missing_provider_rows() -> None:
    trajectory = prepare_provider_spatial_trajectory(
        authorities=TrajectoryAuthorityIdentities(
            recording_id="recording-001",
            provider_id="detection_bbox_centroid.v1",
            estimator_id="detector-run-001",
            source_id="track-source-001",
            timing_authority_id="camera-clock.v1",
            timeline_authority_id="recording-timeline.v1",
            coordinate_authority_id="camera-native-pixels.v2",
            selection_authority_id="selection-canary.v1",
        ),
        rows=ProviderTrackSamples(
            track_sample_key=[[1, 0], [1, 2]],
            acquisition_frame=[0, 2],
            subject_identity=["fish-1", "fish-1"],
            track_identity=["track-1", "track-1"],
            source_position_xy=[[0.25, 0.25], [1.25, 1.25]],
            provider_present=[True, True],
            provider_valid=[True, True],
            provider_reason_code=["ok", "ok"],
        ),
        selection=SelectedFrameMembership(
            recording_id="recording-001",
            timeline_authority_id="recording-timeline.v1",
            selection_authority_id="selection-canary.v1",
            acquisition_frames=[0, 1, 2],
            membership_keys=["pre-0", "pre-1", "chaser-2"],
            occurrence_ids=["pre", "pre", "chaser"],
            roles=["baseline", "baseline", "treatment"],
        ),
        transform=SourceCameraToArenaMMTransform(
            source_coordinate_authority_id="camera-native-pixels.v2",
            target_coordinate_authority_id="arena-mm.v1",
            matrix=np.eye(3),
            grid_extent_mm=(0.0, 2.0, 0.0, 2.0),
        ),
    )
    result = calculate_provider_occupancy_v2(
        occupancy_samples_from_trajectory(trajectory),
        OccupancyGrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        OccupancyTimingPolicy(100.0),
    )

    assert result.pooled.expected_selected_frames == 3
    assert result.pooled.provider_present_count == 2
    assert result.pooled.provider_missing_count == 1
    assert [item.expected_selected_frames for item in result.per_occurrence] == [2, 1]
    assert [item.provider_present_count for item in result.per_occurrence] == [1, 1]


def test_compiled_overlap_is_counted_once_pooled_and_in_each_occurrence() -> None:
    authority = TimelineAuthority(
        recording_id="recording-001",
        timeline_id="recording-timeline.v1",
        stimulus_authority_id="stimulus-run.v1",
        stimulus_authority_sha256="a" * 64,
        acquisition_frame_domain="camera_acquisition_frame_index",
        acquisition_frame_count=5,
        source_video_metadata_ref="source-video-metadata.v2",
        source_video_metadata_sha256="b" * 64,
        acquisition_clock_authority_ref="camera-clock.v1",
        acquisition_clock_authority_sha256="c" * 64,
        source_metadata_sha256="d" * 64,
    )
    first = stimulus_step_reference(
        reference_id="step-a",
        label="A",
        start_frame=0,
        end_frame=3,
        authority=authority,
        occurrence_id="occurrence-a",
    )
    second = stimulus_step_reference(
        reference_id="step-b",
        label="B",
        start_frame=2,
        end_frame=5,
        authority=authority,
        occurrence_id="occurrence-b",
    )
    compiled = compile_selection(
        SelectionSpec(
            selection_id="overlap-selection.v1",
            expression=union(
                member(first, role=RoleMetadata("baseline")),
                member(second, role=RoleMetadata("treatment")),
            ),
            aggregation_policy="keep_occurrences",
        )
    )
    selection = selected_frame_membership_from_compiled_selection(compiled)
    assert selection.occurrence_ids[2] == ("occurrence-a", "occurrence-b")
    trajectory = prepare_provider_spatial_trajectory(
        authorities=TrajectoryAuthorityIdentities(
            recording_id="recording-001",
            provider_id="detection_bbox_centroid.v1",
            estimator_id="detector-run-001",
            source_id="track-source-001",
            timing_authority_id="camera-clock.v1",
            timeline_authority_id="recording-timeline.v1",
            coordinate_authority_id="camera-native-pixels.v2",
            selection_authority_id=compiled.resolved_digest,
        ),
        rows=ProviderTrackSamples(
            track_sample_key=[[1, frame] for frame in range(5)],
            acquisition_frame=list(range(5)),
            subject_identity=["fish-1"] * 5,
            track_identity=["track-1"] * 5,
            source_position_xy=[[0.25 + frame * 0.1, 0.25] for frame in range(5)],
            provider_present=[True] * 5,
            provider_valid=[True] * 5,
            provider_reason_code=["ok"] * 5,
        ),
        selection=selection,
        transform=SourceCameraToArenaMMTransform(
            source_coordinate_authority_id="camera-native-pixels.v2",
            target_coordinate_authority_id="arena-mm.v1",
            matrix=np.eye(3),
            grid_extent_mm=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    result = calculate_provider_occupancy_v2(
        occupancy_samples_from_trajectory(trajectory),
        OccupancyGrid([0.0, 1.0], [0.0, 1.0]),
        OccupancyTimingPolicy(100.0),
    )

    assert result.pooled.expected_selected_frames == 5
    assert result.pooled.valid_in_grid_sample_count == 5
    assert [item.expected_selected_frames for item in result.per_occurrence] == [3, 3]
    assert [item.valid_in_grid_sample_count for item in result.per_occurrence] == [3, 3]
