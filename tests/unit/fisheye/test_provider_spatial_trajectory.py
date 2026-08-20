from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis_workflows.composable_stimulus_selection import (
    RoleMetadata,
    SelectionSpec,
    TimelineAuthority,
    compile_selection,
    member,
    stimulus_step_reference,
    union,
)
from fisheye.analysis.provider_spatial_trajectory import (
    ProviderSpatialTrajectoryError,
    ProviderTrackSamples,
    SelectedFrameMembership,
    SourceCameraToArenaMMTransform,
    TrajectoryAuthorityIdentities,
    prepare_provider_spatial_trajectory,
    prepare_provider_track_sample_trajectory,
    selected_frame_membership_from_compiled_selection,
)


def _authorities() -> TrajectoryAuthorityIdentities:
    return TrajectoryAuthorityIdentities(
        recording_id="recording-001",
        provider_id="detection_bbox_centroid.v1",
        track_sample_policy_id="observation_to_track.single_subject.v1",
        estimator_id="detector-run-001",
        source_id="raw-track-source-001",
        timing_authority_id="camera-clock.v1",
        timeline_authority_id="recording-timeline.v1",
        coordinate_authority_id="camera-native-pixels.v2",
        selection_authority_id="selection-canary.v1",
    )


def _selection() -> SelectedFrameMembership:
    return SelectedFrameMembership(
        recording_id="recording-001",
        timeline_authority_id="recording-timeline.v1",
        selection_authority_id="selection-canary.v1",
        acquisition_frames=[0, 1, 2, 4],
        membership_keys=["before-0", "before-1", "chaser-2", "chaser-4"],
        occurrence_ids=["before", "before", "chaser", "chaser"],
        roles=["black_before", "black_before", "chaser", "chaser"],
    )


def _transform(
    *,
    matrix: object | None = None,
    grid_extent_mm: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0),
    source_camera_extent_px: tuple[float, float, float, float] | None = (
        0.0,
        200.0,
        0.0,
        200.0,
    ),
) -> SourceCameraToArenaMMTransform:
    return SourceCameraToArenaMMTransform(
        source_coordinate_authority_id="camera-native-pixels.v2",
        target_coordinate_authority_id="arena-mm.goodbatbadbat.v1",
        matrix=(
            np.eye(3, dtype=np.float64)
            if matrix is None
            else matrix
        ),
        grid_extent_mm=grid_extent_mm,
        source_camera_extent_px=source_camera_extent_px,
    )


def _rows() -> ProviderTrackSamples:
    return ProviderTrackSamples(
        track_sample_key=[[1, 0], [1, 1], [1, 2], [1, 3]],
        acquisition_frame=[0, 1, 2, 3],
        subject_identity=["fish-1"] * 4,
        track_identity=["track-1"] * 4,
        source_position_xy=np.array(
            [[1.0, 2.0], [np.nan, np.nan], [2.0, 3.0], [150.0, 1.0]],
            dtype=np.float64,
        ),
        provider_present=[True, False, True, True],
        provider_valid=[True, False, False, True],
        provider_reason_code=["ok", "provider_missing", "provider_invalid", "ok"],
    )


def test_preparation_preserves_rows_and_separates_all_states() -> None:
    result = prepare_provider_spatial_trajectory(
        authorities=_authorities(),
        rows=_rows(),
        selection=_selection(),
        transform=_transform(
            matrix=np.array(
                [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]
            )
        ),
    )

    np.testing.assert_array_equal(result.track_sample_key, _rows().track_sample_key)
    np.testing.assert_array_equal(result.source_row_index, [0, 1, 2, 3])
    np.testing.assert_array_equal(result.acquisition_frame, [0, 1, 2, 3])
    np.testing.assert_array_equal(result.in_selection, [True, True, True, False])
    np.testing.assert_array_equal(result.provider_present, [True, False, True, True])
    np.testing.assert_array_equal(result.provider_valid, [True, False, False, True])
    np.testing.assert_array_equal(result.source_position_valid, [True, False, False, True])
    np.testing.assert_array_equal(result.source_extent_valid, [True, False, True, True])
    np.testing.assert_array_equal(result.transform_valid, [True, False, False, True])
    np.testing.assert_array_equal(result.in_grid, [True, False, False, False])
    np.testing.assert_allclose(result.arena_position_xy[0], [12.0, 24.0])
    assert np.isnan(result.arena_position_xy[1:3]).all()
    np.testing.assert_allclose(result.arena_position_xy[3], [310.0, 22.0])
    assert result.selection_membership_key == (
        ("before-0",),
        ("before-1",),
        ("chaser-2",),
        (),
    )
    assert result.selection_role == (
        ("black_before",),
        ("black_before",),
        ("chaser",),
        (),
    )
    assert result.reason_codes[0] == ("ok",)
    assert result.reason_codes[1] == ("provider_missing",)
    assert result.reason_codes[2] == ("provider_invalid",)
    assert result.reason_codes[3] == ("not_in_selection", "out_of_grid")
    assert result.counts.expected_selected_frames == 4
    assert result.counts.source_rows == 4
    assert result.counts.selected_source_rows == 3
    assert result.counts.missing_selected_frames == 1
    assert result.counts.valid_position_rows == 2
    assert result.counts.transform_valid_rows == 2
    assert result.counts.in_grid_rows == 1
    assert result.counts.selected_valid_position_rows == 1
    assert result.counts.selected_transform_valid_rows == 1
    assert result.counts.selected_in_grid_rows == 1
    assert result.counts.selected_missing_provider_rows == 1
    assert result.counts.selected_invalid_provider_rows == 1
    assert result.reason_counts["provider_missing"] == 1
    assert result.reason_counts["provider_invalid"] == 1
    assert result.reason_counts["out_of_grid"] == 1
    assert result.selected_reason_counts["out_of_grid"] == 0
    assert result.selected_reason_counts["provider_missing"] == 1
    assert result.selected_reason_counts["provider_invalid"] == 1


def test_membership_identity_repeats_across_frames_but_not_within_a_row() -> None:
    selection = SelectedFrameMembership(
        recording_id="recording-001",
        timeline_authority_id="recording-timeline.v1",
        selection_authority_id="selection-canary.v1",
        acquisition_frames=[2, 3],
        membership_keys=[["membership:source-membership-digest"]] * 2,
        occurrence_ids=[["occurrence-1"]] * 2,
        roles=[["treatment"]] * 2,
    )
    assert selection.membership_keys == (
        ("membership:source-membership-digest",),
        ("membership:source-membership-digest",),
    )
    with pytest.raises(ProviderSpatialTrajectoryError, match="duplicate identities"):
        SelectedFrameMembership(
            recording_id="recording-001",
            timeline_authority_id="recording-timeline.v1",
            selection_authority_id="selection-canary.v1",
            acquisition_frames=[2],
            membership_keys=[["member-a", "member-a"]],
            occurrence_ids=[["occurrence-1", "occurrence-1"]],
            roles=[["treatment", "treatment"]],
        )


def test_compiled_overlap_keeps_stable_memberships_and_unique_pooled_frames() -> None:
    authority = TimelineAuthority(
        recording_id="recording-001",
        timeline_id="recording-timeline.v1",
        stimulus_authority_id="stimulus-run.v1",
        stimulus_authority_sha256="a" * 64,
        acquisition_frame_domain="camera_acquisition_frame_index",
        acquisition_frame_count=5,
        source_video_metadata_ref="source-video-metadata.v1",
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

    assert selection.acquisition_frames.tolist() == [0, 1, 2, 3, 4]
    assert len(set(selection.acquisition_frames.tolist())) == 5
    first_key = selection.membership_keys[0][0]
    second_key = selection.membership_keys[2][1]
    assert first_key == selection.membership_keys[1][0]
    assert first_key == selection.membership_keys[2][0]
    assert second_key == selection.membership_keys[2][1]
    assert second_key == selection.membership_keys[3][0]
    assert all(key.startswith("membership:") and key.count(":") == 1
               for row in selection.membership_keys for key in row)


def test_source_camera_extent_is_half_open_and_blocks_transform() -> None:
    rows = ProviderTrackSamples(
        track_sample_key=[[1, frame] for frame in range(5)],
        acquisition_frame=list(range(5)),
        subject_identity=["fish-1"] * 5,
        track_identity=["track-1"] * 5,
        source_position_xy=[
            [99.999, 50.0],
            [100.0, 50.0],
            [50.0, 100.0],
            [50.0, 99.999],
            [150.0, 50.0],
        ],
        provider_present=[True] * 5,
        provider_valid=[True] * 5,
        provider_reason_code=["ok"] * 5,
    )
    selection = SelectedFrameMembership(
        recording_id="recording-001",
        timeline_authority_id="recording-timeline.v1",
        selection_authority_id="selection-canary.v1",
        acquisition_frames=list(range(5)),
        membership_keys=[["member"]] * 5,
        occurrence_ids=[["occurrence"]] * 5,
        roles=[["treatment"]] * 5,
    )
    result = prepare_provider_spatial_trajectory(
        authorities=_authorities(),
        rows=rows,
        selection=selection,
        transform=_transform(
            matrix=np.eye(3),
            grid_extent_mm=(0.0, 200.0, 0.0, 200.0),
            source_camera_extent_px=(0.0, 100.0, 0.0, 100.0),
        ),
    )

    np.testing.assert_array_equal(result.source_extent_valid, [True, False, False, True, False])
    np.testing.assert_array_equal(result.transform_valid, [True, False, False, True, False])
    np.testing.assert_array_equal(result.in_grid, [True, False, False, True, False])
    assert result.reason_codes[1] == ("source_position_out_of_extent",)
    assert result.reason_codes[2] == ("source_position_out_of_extent",)
    assert result.reason_codes[4] == ("source_position_out_of_extent",)
    assert result.counts.source_extent_valid_rows == 2
    assert result.counts.source_position_out_of_extent_rows == 3
    assert result.counts.transform_invalid_rows == 0
    assert np.isnan(result.arena_position_xy[[1, 2, 4]]).all()


def test_missing_source_camera_extent_fails_closed() -> None:
    with pytest.raises(ProviderSpatialTrajectoryError, match="source_camera_extent_px"):
        prepare_provider_spatial_trajectory(
            authorities=_authorities(),
            rows=_rows(),
            selection=_selection(),
            transform=_transform(source_camera_extent_px=None),
        )


def test_selection_is_a_frame_lookup_not_a_same_length_join() -> None:
    rows = ProviderTrackSamples(
        track_sample_key=[[1, 0], [1, 2], [1, 10]],
        acquisition_frame=[0, 2, 10],
        subject_identity=["fish-1"] * 3,
        track_identity=["track-1"] * 3,
        source_position_xy=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        provider_present=[True] * 3,
        provider_valid=[True] * 3,
        provider_reason_code=["ok"] * 3,
    )
    selection = SelectedFrameMembership(
        recording_id="recording-001",
        timeline_authority_id="recording-timeline.v1",
        selection_authority_id="selection-canary.v1",
        acquisition_frames=[2, 10],
        membership_keys=["chaser-2", "chaser-10"],
        occurrence_ids=["chaser", "chaser"],
        roles=["chaser", "chaser"],
    )

    result = prepare_provider_track_sample_trajectory(
        authorities=_authorities(),
        rows=rows,
        selection=selection,
        transform=_transform(),
    )

    np.testing.assert_array_equal(result.in_selection, [False, True, True])
    assert result.counts.expected_selected_frames == 2
    assert result.counts.selected_source_rows == 2
    assert result.counts.missing_selected_frames == 0


def test_boundary_of_grid_is_inclusive() -> None:
    rows = ProviderTrackSamples(
        track_sample_key=[[1, 0]],
        acquisition_frame=[0],
        subject_identity=["fish-1"],
        track_identity=["track-1"],
        source_position_xy=[[10.0, 20.0]],
        provider_present=[True],
        provider_valid=[True],
        provider_reason_code=["ok"],
    )
    selection = SelectedFrameMembership(
        recording_id="recording-001",
        timeline_authority_id="recording-timeline.v1",
        selection_authority_id="selection-canary.v1",
        acquisition_frames=[0],
        membership_keys=["member-0"],
        occurrence_ids=["occurrence-0"],
        roles=["chaser"],
    )

    result = prepare_provider_spatial_trajectory(
        authorities=_authorities(),
        rows=rows,
        selection=selection,
        transform=_transform(grid_extent_mm=(0.0, 10.0, 0.0, 20.0)),
    )

    assert bool(result.transform_valid[0]) is True
    assert bool(result.in_grid[0]) is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("track_sample_key", [[1, 0], [1, 0]], "duplicates"),
        ("track_sample_key", [[1, 1], [1, 0]], "reordered"),
    ],
)
def test_duplicate_or_reordered_source_rows_fail_closed(
    field: str, value: object, message: str
) -> None:
    kwargs = {
        "track_sample_key": [[1, 0], [1, 1]],
        "acquisition_frame": [0, 1],
        "subject_identity": ["fish-1", "fish-1"],
        "track_identity": ["track-1", "track-1"],
        "source_position_xy": [[1.0, 1.0], [2.0, 2.0]],
        "provider_present": [True, True],
        "provider_valid": [True, True],
        "provider_reason_code": ["ok", "ok"],
    }
    kwargs[field] = value
    if field == "track_sample_key":
        kwargs["acquisition_frame"] = [row[1] for row in value]
    with pytest.raises(ProviderSpatialTrajectoryError, match=message):
        ProviderTrackSamples(**kwargs)


def test_single_subject_profile_rejects_two_samples_for_one_frame() -> None:
    with pytest.raises(ProviderSpatialTrajectoryError, match="more than one track sample"):
        ProviderTrackSamples(
            track_sample_key=[[1, 0], [2, 0]],
            acquisition_frame=[0, 0],
            subject_identity=["fish-1", "fish-1"],
            track_identity=["track-1", "track-2"],
            source_position_xy=[[1.0, 1.0], [2.0, 2.0]],
            provider_present=[True, True],
            provider_valid=[True, True],
            provider_reason_code=["ok", "ok"],
        )


def test_cardinality_and_mixed_authorities_fail_closed() -> None:
    with pytest.raises(ProviderSpatialTrajectoryError, match="mismatched cardinality"):
        ProviderTrackSamples(
            track_sample_key=[[1, 0]],
            acquisition_frame=[0, 1],
            subject_identity=["fish-1"],
            track_identity=["track-1"],
            source_position_xy=[[1.0, 1.0]],
            provider_present=[True],
            provider_valid=[True],
            provider_reason_code=["ok"],
        )

    rows = ProviderTrackSamples(
        track_sample_key=[[1, 0], [1, 1]],
        acquisition_frame=[0, 1],
        subject_identity=["fish-1", "fish-1"],
        track_identity=["track-1", "track-1"],
        source_position_xy=[[1.0, 1.0], [2.0, 2.0]],
        provider_present=[True, True],
        provider_valid=[True, True],
        provider_reason_code=["ok", "ok"],
        recording_ids=["recording-001", "recording-002"],
    )
    with pytest.raises(ProviderSpatialTrajectoryError, match="mixed recording"):
        prepare_provider_spatial_trajectory(
            authorities=_authorities(),
            rows=rows,
            selection=_selection(),
            transform=_transform(),
        )


def test_authority_selection_and_transform_mismatches_fail_closed() -> None:
    with pytest.raises(ProviderSpatialTrajectoryError, match="stale/unknown"):
        TrajectoryAuthorityIdentities(
            recording_id="recording-001",
            provider_id="latest",
            track_sample_policy_id="observation_to_track.single_subject.v1",
            estimator_id="estimator-1",
            source_id="source-1",
            timing_authority_id="timing-1",
            timeline_authority_id="timeline-1",
            coordinate_authority_id="coordinate-1",
            selection_authority_id="selection-1",
        )
    with pytest.raises(ProviderSpatialTrajectoryError, match="Transform coordinate"):
        prepare_provider_spatial_trajectory(
            authorities=_authorities(),
            rows=_rows(),
            selection=_selection(),
            transform=SourceCameraToArenaMMTransform(
                source_coordinate_authority_id="other-camera.v1",
                target_coordinate_authority_id="arena-mm.v1",
                matrix=np.eye(3),
                grid_extent_mm=(0.0, 100.0, 0.0, 100.0),
                source_camera_extent_px=(0.0, 200.0, 0.0, 200.0),
            ),
        )


def test_invalid_transform_and_extent_fail_closed() -> None:
    with pytest.raises(ProviderSpatialTrajectoryError, match="non-singular"):
        SourceCameraToArenaMMTransform(
            source_coordinate_authority_id="camera.v1",
            target_coordinate_authority_id="arena.v1",
            matrix=np.zeros((3, 3)),
            grid_extent_mm=(0.0, 1.0, 0.0, 1.0),
        )
    with pytest.raises(ProviderSpatialTrajectoryError, match="increasing"):
        SourceCameraToArenaMMTransform(
            source_coordinate_authority_id="camera.v1",
            target_coordinate_authority_id="arena.v1",
            matrix=np.eye(3),
            grid_extent_mm=(1.0, 0.0, 0.0, 1.0),
        )


def test_result_is_read_only_and_record_is_digest_bound() -> None:
    result = prepare_provider_spatial_trajectory(
        authorities=_authorities(),
        rows=_rows(),
        selection=_selection(),
        transform=_transform(),
    )

    with pytest.raises(ValueError):
        result.arena_position_xy[0, 0] = 99.0
    record = result.as_record()
    assert record["schema_id"] == "palette.provider_spatial_trajectory"
    assert record["row_axis"] == "track_samples"
    assert record["smoothing"] == "none"
    assert record["fallback"] == "none"
    assert (
        record["authorities"]["track_sample_policy_id"]
        == "observation_to_track.single_subject.v1"
    )
    assert record["selection"]["sha256"] == _selection().sha256
    assert record["transform"]["sha256"] == _transform().sha256
    assert len(result.trajectory_sha256) == 64
