from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.provider_spatial_trajectory import (
    ProviderSpatialTrajectoryError,
    ProviderTrackSamples,
    SelectedFrameMembership,
    SourceCameraToArenaMMTransform,
    TrajectoryAuthorityIdentities,
    prepare_provider_spatial_trajectory,
    prepare_provider_track_sample_trajectory,
)


def _authorities() -> TrajectoryAuthorityIdentities:
    return TrajectoryAuthorityIdentities(
        recording_id="recording-001",
        provider_id="detection_bbox_centroid.v1",
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
    assert record["selection"]["sha256"] == _selection().sha256
    assert record["transform"]["sha256"] == _transform().sha256
    assert len(result.trajectory_sha256) == 64
