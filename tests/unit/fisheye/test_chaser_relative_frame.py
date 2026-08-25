from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_relative_frame import (
    ACTIVE_ORTHOGONAL_POSITION_VALIDITY_POLICY,
    AcquisitionFrameKeys,
    BodyFrameInput,
    ChaserObservations,
    ChaserRelativeFrameError,
    ChaserRelativeFrameInput,
    CoordinatePolicy,
    ProviderSourceAuthority,
    ScalePolicy,
    TimingPolicy,
    compute_chaser_relative_frame,
)


def _keys(
    *,
    frame_ids: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    recording_id: str = "recording-001",
    timestamp_ns: np.ndarray | None = None,
    include_timestamps: bool = True,
) -> AcquisitionFrameKeys:
    return AcquisitionFrameKeys(
        recording_id=recording_id,
        acquisition_frame_id=(
            np.arange(4, dtype=np.int64) if frame_ids is None else frame_ids
        ),
        track_sample_id=(
            np.arange(100, 104, dtype=np.int64) if sample_ids is None else sample_ids
        ),
        row_axis_authority_id="camera-row-axis-v1",
        row_axis_authority_digest="row-axis-digest",
        timestamp_ns=(
            None
            if not include_timestamps
            else (
                np.arange(4, dtype=np.int64) * 10_000_000
                if timestamp_ns is None
                else timestamp_ns
            )
        ),
    )


def _authority(
    *,
    provider_id: str,
    provider_digest: str,
    source_authority_id: str,
    source_digest: str,
    recording_id: str = "recording-001",
    coordinate_authority_id: str = "source-camera-coordinates-v1",
    timing_authority_id: str = "camera-timing-v1",
    row_axis_authority_id: str = "camera-row-axis-v1",
    row_axis_authority_digest: str = "row-axis-digest",
) -> ProviderSourceAuthority:
    return ProviderSourceAuthority(
        recording_id=recording_id,
        source_authority_id=source_authority_id,
        source_digest=source_digest,
        provider_id=provider_id,
        provider_digest=provider_digest,
        coordinate_authority_id=coordinate_authority_id,
        scale_authority_id="camera-scale-v1",
        timing_authority_id=timing_authority_id,
        row_axis_authority_id=row_axis_authority_id,
        row_axis_authority_digest=row_axis_authority_digest,
    )


def _base_input(
    *,
    frame_keys: AcquisitionFrameKeys | None = None,
    fish_xy: np.ndarray | None = None,
    fish_valid: np.ndarray | None = None,
    chaser_xy: np.ndarray | None = None,
    chaser_valid: np.ndarray | None = None,
    identities: tuple[str, ...] = ("chaser-0", "chaser-1"),
    behavior_roles: np.ndarray | None = None,
    selection: np.ndarray | None = None,
    occurrence: np.ndarray | None = None,
    active: np.ndarray | None = None,
    trial_ids: np.ndarray | None = None,
    fish_authority: ProviderSourceAuthority | None = None,
    chaser_authority: ProviderSourceAuthority | None = None,
    body_frame: BodyFrameInput | None = None,
) -> ChaserRelativeFrameInput:
    keys = _keys() if frame_keys is None else frame_keys
    n = keys.row_count
    m = len(identities)
    if fish_xy is None:
        fish_xy = np.zeros((n, 2), dtype=np.float64)
    if fish_valid is None:
        fish_valid = np.ones(n, dtype=bool)
    if chaser_xy is None:
        chaser_xy = np.zeros((n, m, 2), dtype=np.float64)
        chaser_xy[:, 0, 0] = 3.0
        if m > 1:
            chaser_xy[:, 1, 0] = 4.0
    if chaser_valid is None:
        chaser_valid = np.ones((n, m), dtype=bool)
    if selection is None:
        selection = np.ones(n, dtype=bool)
    if occurrence is None:
        occurrence = np.ones((n, m), dtype=bool)
    return ChaserRelativeFrameInput(
        frame_keys=keys,
        fish_xy=fish_xy,
        fish_valid=fish_valid,
        fish_source_row_index=np.arange(n, dtype=np.int64),
        fish_authority=(
            _authority(
                provider_id="fish-position-provider",
                provider_digest="fish-provider-digest",
                source_authority_id="fish-source",
                source_digest="fish-source-digest",
            )
            if fish_authority is None
            else fish_authority
        ),
        chasers=ChaserObservations(
            identities=identities,
            behavior_roles=(
                np.tile(
                    np.asarray([f"role-{index}" for index in range(m)], dtype="<U16"),
                    (n, 1),
                )
                if behavior_roles is None
                else behavior_roles
            ),
            xy=chaser_xy,
            valid=chaser_valid,
            source_row_index=np.arange(n * m, dtype=np.int64).reshape(n, m),
            authority=(
                _authority(
                    provider_id="chaser-state-provider",
                    provider_digest="chaser-provider-digest",
                    source_authority_id="chaser-source",
                    source_digest="chaser-source-digest",
                )
                if chaser_authority is None
                else chaser_authority
            ),
            trial_ids=trial_ids,
            active=active,
        ),
        selection_membership=selection,
        occurrence_membership=occurrence,
        coordinate_policy=CoordinatePolicy(
            coordinate_authority_id="source-camera-coordinates-v1",
            coordinate_frame="source_camera_pixels",
        ),
        scale_policy=ScalePolicy(
            scale_authority_id="camera-scale-v1",
            scale_digest="camera-scale-digest",
            pixels_per_unit=10.0,
        ),
        timing_policy=TimingPolicy(
            timing_authority_id="camera-timing-v1",
            timing_digest="camera-timing-digest",
            recording_id=keys.recording_id,
        ),
        body_frame=body_frame,
    )


def _body(
    keys: AcquisitionFrameKeys,
    *,
    origin: np.ndarray | None = None,
    forward: np.ndarray | None = None,
    left: np.ndarray | None = None,
    axis_valid: np.ndarray | None = None,
    authority: ProviderSourceAuthority | None = None,
) -> BodyFrameInput:
    n = keys.row_count
    return BodyFrameInput(
        frame_keys=keys,
        origin_xy=(np.zeros((n, 2), dtype=np.float64) if origin is None else origin),
        forward_axis_xy=(
            np.repeat(np.asarray([[1.0, 0.0]]), n, axis=0)
            if forward is None
            else forward
        ),
        left_axis_xy=(
            np.repeat(np.asarray([[0.0, -1.0]]), n, axis=0)
            if left is None
            else left
        ),
        axis_valid=(np.ones(n, dtype=bool) if axis_valid is None else axis_valid),
        source_row_index=np.arange(n, dtype=np.int64),
        authority=(
            _authority(
                provider_id="body-frame-provider",
                provider_digest="body-provider-digest",
                source_authority_id="body-frame-source",
                source_digest="body-frame-source-digest",
            )
            if authority is None
            else authority
        ),
    )


def test_distance_and_complete_chaser_axis_are_derived_in_memory() -> None:
    result = compute_chaser_relative_frame(_base_input())

    assert result.relative_xy.shape == (4, 2, 2)
    assert result.chaser_identities == ("chaser-0", "chaser-1")
    np.testing.assert_allclose(result.distance_px[:, 0], 3.0)
    np.testing.assert_allclose(result.distance_px[:, 1], 4.0)
    np.testing.assert_allclose(result.distance_physical[:, 0], 0.3)
    np.testing.assert_allclose(result.relative_xy_physical[:, 1, 0], 0.4)
    assert result.relative_valid.all()
    assert result.fish_transition_valid.tolist() == [False, True, True, True]
    assert result.relative_transition_valid[1:].all()
    np.testing.assert_array_equal(
        result.timestamp_delta_ns,
        [-1, 10_000_000, 10_000_000, 10_000_000],
    )
    assert result.relative_reason_code.tolist() == [["valid", "valid"]] * 4
    assert not result.relative_xy.flags.writeable
    with pytest.raises(ValueError):
        result.distance_px[0, 0] = 99.0


def test_nearest_tie_chooses_lowest_chaser_index_and_identity() -> None:
    xy = np.zeros((4, 3, 2), dtype=np.float64)
    xy[:, 0, 0] = 2.0
    xy[:, 1, 0] = -2.0
    xy[:, 2, 0] = 5.0
    result = compute_chaser_relative_frame(
        _base_input(chaser_xy=xy, identities=("a", "b", "c"))
    )

    assert result.nearest_chaser_index.tolist() == [0, 0, 0, 0]
    assert result.nearest_chaser_identity == ("a",) * 4


def test_invalid_selection_occurrence_active_and_coordinates_have_reason_codes() -> None:
    fish_valid = np.ones(4, dtype=bool)
    fish_valid[1] = False
    chaser_valid = np.ones((4, 2), dtype=bool)
    chaser_valid[2, 0] = False
    chaser_xy = np.zeros((4, 2, 2), dtype=np.float64)
    chaser_xy[:, :, 0] = 2.0
    chaser_xy[3, 0, 0] = np.nan
    selection = np.ones(4, dtype=bool)
    selection[0] = False
    occurrence = np.ones((4, 2), dtype=bool)
    occurrence[1, 1] = False
    active = np.ones((4, 2), dtype=bool)
    active[2, 1] = False
    result = compute_chaser_relative_frame(
        _base_input(
            fish_valid=fish_valid,
            chaser_xy=chaser_xy,
            chaser_valid=chaser_valid,
            selection=selection,
            occurrence=occurrence,
            active=active,
        )
    )

    assert result.relative_reason_code[0].tolist() == ["selection_excluded"] * 2
    assert result.relative_reason_code[1].tolist() == ["fish_invalid", "occurrence_excluded"]
    assert result.relative_reason_code[2].tolist() == ["chaser_invalid", "valid"]
    assert result.relative_reason_code[3, 0] == "nonfinite_coordinate"
    assert np.isnan(result.distance_px[3, 0])


def test_transition_censoring_separates_frame_time_selection_occurrence_and_trial_boundaries() -> None:
    keys = _keys(
        frame_ids=np.asarray([0, 1, 3, 4], dtype=np.int64),
        timestamp_ns=np.asarray([0, 10, 20, 20], dtype=np.int64),
    )
    selection = np.asarray([True, True, True, False], dtype=bool)
    occurrence = np.ones((4, 2), dtype=bool)
    occurrence[1, 1] = False
    trials = np.asarray([[1, 1], [1, 1], [2, 1], [2, 1]], dtype=np.int64)
    result = compute_chaser_relative_frame(
        _base_input(
            frame_keys=keys,
            selection=selection,
            occurrence=occurrence,
            trial_ids=trials,
        )
    )

    assert result.relative_transition_reason_code[1, 0] == "valid"
    assert result.relative_transition_reason_code[1, 1] == "occurrence_boundary"
    assert result.relative_transition_reason_code[2, 0] == (
        "nonconsecutive_acquisition_frame"
    )
    assert result.relative_transition_reason_code[3, 0] == "nonpositive_timestamp_delta"


def test_missing_camera_timestamps_explicitly_censors_transitions() -> None:
    result = compute_chaser_relative_frame(
        _base_input(frame_keys=_keys(include_timestamps=False))
    )

    assert not result.fish_transition_valid.any()
    assert result.fish_transition_reason_code.tolist() == [
        "no_predecessor",
        "timestamp_unavailable",
        "timestamp_unavailable",
        "timestamp_unavailable",
    ]


def test_optional_trial_ids_and_active_state_are_preserved() -> None:
    trial_ids = np.asarray([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=np.int64)
    active = np.asarray([[True, True], [True, False], [True, True], [False, True]])
    result = compute_chaser_relative_frame(
        _base_input(trial_ids=trial_ids, active=active)
    )

    np.testing.assert_array_equal(result.chaser_trial_ids, trial_ids)
    np.testing.assert_array_equal(result.chaser_active, active)


def test_controller_activity_can_be_orthogonal_to_position_validity() -> None:
    active = np.asarray(
        [[False, False], [True, False], [False, True], [False, False]],
        dtype=bool,
    )
    result = compute_chaser_relative_frame(
        _base_input(
            active=active,
        )
    )

    assert result.relative_valid.all()
    assert np.isfinite(result.distance_physical).all()
    np.testing.assert_array_equal(result.chaser_active, active)
    assert (
        result.active_position_validity_policy
        == ACTIVE_ORTHOGONAL_POSITION_VALIDITY_POLICY
    )


def test_body_frame_bearing_quadrants_use_y_down_axes_without_flip() -> None:
    xy = np.zeros((4, 4, 2), dtype=np.float64)
    xy[:, 0] = [1.0, 0.0]
    xy[:, 1] = [0.0, -1.0]
    xy[:, 2] = [-1.0, 0.0]
    xy[:, 3] = [0.0, 1.0]
    keys = _keys()
    result = compute_chaser_relative_frame(
        _base_input(
            frame_keys=keys,
            chaser_xy=xy,
            identities=("chaser-0", "chaser-1", "chaser-2", "chaser-3"),
            behavior_roles=np.tile(
                np.asarray(["front", "left", "behind", "right"], dtype="<U8"),
                (4, 1),
            ),
            body_frame=_body(keys),
        )
    )

    np.testing.assert_allclose(result.egocentric_bearing_deg[0], [0.0, 90.0, 180.0, -90.0])
    assert result.egocentric_valid[0].tolist() == [True] * 4
    np.testing.assert_allclose(result.forward_coordinate_px[0], [1.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(result.left_coordinate_px[0], [0.0, 1.0, 0.0, -1.0])
    np.testing.assert_allclose(result.body_frame_heading_deg, 0.0)
    assert result.heading_transition_valid.tolist() == [False, True, True, True]


def test_identity_and_time_varying_behavior_role_are_separate() -> None:
    roles = np.asarray(
        [
            ["aggressive", "inert"],
            ["aggressive", "inert"],
            ["inactive", "inert"],
            ["inactive", "inert"],
        ],
        dtype="<U16",
    )
    result = compute_chaser_relative_frame(
        _base_input(identities=("chaser-0", "chaser-1"), behavior_roles=roles)
    )

    assert result.chaser_identities == ("chaser-0", "chaser-1")
    np.testing.assert_array_equal(result.chaser_behavior_roles, roles)


def test_body_frame_origin_not_position_provider_origin_drives_bearing() -> None:
    keys = _keys()
    body = _body(keys)
    body = replace(
        body,
        origin_xy=np.repeat(np.asarray([[10.0, 0.0]]), 4, axis=0),
    )
    fish_xy = np.repeat(np.asarray([[100.0, 100.0]]), 4, axis=0)
    chaser_xy = np.zeros((4, 2, 2), dtype=np.float64)
    chaser_xy[:, 0] = [11.0, 0.0]
    chaser_xy[:, 1] = [10.0, -1.0]

    result = compute_chaser_relative_frame(
        _base_input(
            frame_keys=keys,
            fish_xy=fish_xy,
            chaser_xy=chaser_xy,
            identities=("chaser-0", "chaser-1"),
            body_frame=body,
        )
    )

    np.testing.assert_allclose(result.egocentric_bearing_deg[0], [0.0, 90.0])
    np.testing.assert_allclose(result.body_relative_xy[0], [[1.0, 0.0], [0.0, -1.0]])
    assert np.all(result.distance_px > 100.0)


def test_missing_optional_body_frame_is_explicit_and_does_not_block_distance() -> None:
    result = compute_chaser_relative_frame(_base_input())

    assert result.body_frame_present is False
    assert not result.body_frame_valid.any()
    assert np.isnan(result.egocentric_bearing_deg).all()
    assert result.egocentric_reason_code[0, 0] == "body_frame_unavailable"
    assert result.relative_valid.all()


def test_duplicate_acquisition_frame_key_fails_closed() -> None:
    with pytest.raises(ChaserRelativeFrameError, match="duplicate keys"):
        _keys(frame_ids=np.asarray([0, 1, 1, 3], dtype=np.int64))


def test_body_frame_key_order_mismatch_fails_closed() -> None:
    body_keys = _keys(sample_ids=np.asarray([100, 102, 101, 103], dtype=np.int64))
    with pytest.raises(ChaserRelativeFrameError, match="track-sample key/order mismatch"):
        compute_chaser_relative_frame(_base_input(body_frame=_body(body_keys)))


def test_body_frame_same_length_different_acquisition_keys_fails_closed() -> None:
    body_keys = _keys(frame_ids=np.asarray([10, 11, 12, 13], dtype=np.int64))
    with pytest.raises(ChaserRelativeFrameError, match="acquisition-frame key/order mismatch"):
        compute_chaser_relative_frame(_base_input(body_frame=_body(body_keys)))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("row_axis_authority_id", "row_axis_authority_id"),
        ("row_axis_authority_digest", "row_axis_authority_digest"),
    ],
)
def test_provider_authority_requires_exact_row_axis_binding(
    field: str,
    message: str,
) -> None:
    values = {
        "provider_id": "fish-position-provider",
        "provider_digest": "fish-provider-digest",
        "source_authority_id": "fish-source",
        "source_digest": "fish-source-digest",
        field: None,
    }
    with pytest.raises(ChaserRelativeFrameError, match=message):
        _authority(**values)


def test_shape_mismatch_fails_closed_before_computation() -> None:
    with pytest.raises(ChaserRelativeFrameError, match="fish xy must have shape"):
        _base_input(fish_xy=np.zeros((3, 2), dtype=np.float64))


def test_coordinate_authority_mismatch_fails_closed() -> None:
    fish = _authority(
        provider_id="fish-position-provider",
        provider_digest="fish-provider-digest",
        source_authority_id="fish-source",
        source_digest="fish-source-digest",
        coordinate_authority_id="wrong-coordinate-authority",
    )
    with pytest.raises(ChaserRelativeFrameError, match="coordinate-authority mismatch"):
        compute_chaser_relative_frame(_base_input(fish_authority=fish))


def test_recording_and_timing_mismatches_fail_closed() -> None:
    fish = _authority(
        provider_id="fish-position-provider",
        provider_digest="fish-provider-digest",
        source_authority_id="fish-source",
        source_digest="fish-source-digest",
        recording_id="other-recording",
    )
    with pytest.raises(ChaserRelativeFrameError, match="recording authority mismatch"):
        compute_chaser_relative_frame(_base_input(fish_authority=fish))

    timing = TimingPolicy(
        timing_authority_id="wrong-timing-authority",
        timing_digest="camera-timing-digest",
        recording_id="recording-001",
    )
    inputs = _base_input()
    with pytest.raises(ChaserRelativeFrameError, match="timing-authority mismatch"):
        compute_chaser_relative_frame(replace(inputs, timing_policy=timing))


def test_provider_digest_mismatch_for_reused_provider_identity_fails_closed() -> None:
    chaser = _authority(
        provider_id="fish-position-provider",
        provider_digest="changed-provider-digest",
        source_authority_id="chaser-source",
        source_digest="chaser-source-digest",
    )
    with pytest.raises(ChaserRelativeFrameError, match="provider digest mismatch"):
        compute_chaser_relative_frame(_base_input(chaser_authority=chaser))


def test_body_frame_authority_join_mismatch_fails_closed() -> None:
    keys = _keys()
    body_authority = _authority(
        provider_id="body-frame-provider",
        provider_digest="body-provider-digest",
        source_authority_id="body-frame-source",
        source_digest="body-frame-source-digest",
        row_axis_authority_digest="wrong-row-axis-digest",
    )
    with pytest.raises(ChaserRelativeFrameError, match="row-axis authority digest mismatch"):
        compute_chaser_relative_frame(
            _base_input(body_frame=_body(keys, authority=body_authority))
        )


def test_invalid_body_axes_are_explicitly_censored_without_velocity_fallback() -> None:
    keys = _keys()
    axis_valid = np.asarray([True, False, True, True])
    forward = np.repeat(np.asarray([[1.0, 0.0]]), 4, axis=0)
    left = np.repeat(np.asarray([[0.0, -1.0]]), 4, axis=0)
    forward[1] = [np.nan, np.nan]
    left[1] = [np.nan, np.nan]
    origin = np.zeros((4, 2), dtype=np.float64)
    origin[1] = [np.nan, np.nan]
    result = compute_chaser_relative_frame(
        _base_input(
            body_frame=_body(
                keys,
                origin=origin,
                forward=forward,
                left=left,
                axis_valid=axis_valid,
            )
        )
    )

    assert result.egocentric_valid[1].tolist() == [False, False]
    assert result.egocentric_reason_code[1, 0] == "body_frame_invalid"
    assert np.isnan(result.egocentric_bearing_deg[1]).all()
