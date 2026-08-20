from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_relative_distance_view import (
    ChaserRelativeDistanceRegistries,
    ChaserRelativeDistanceView,
    ChaserRelativeDistanceViewError,
    ChaserRelativeDistanceViewInput,
    load_chaser_relative_distance_view,
)
from fisheye.analysis_workflows.chaser_relative_frame import (
    AcquisitionFrameKeys,
    ChaserObservations,
    ChaserRelativeFrameInput,
    CoordinatePolicy,
    ProviderSourceAuthority,
    ScalePolicy,
    TimingPolicy,
    compute_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_relative_frame_storage import (
    ChaserRelativeFramePublicationContext,
    prepare_chaser_relative_frame,
)


_DIGEST = "a" * 64


def _authority(name: str) -> ProviderSourceAuthority:
    return ProviderSourceAuthority(
        recording_id="recording-1",
        source_authority_id=f"{name}-source",
        source_digest=_DIGEST,
        provider_id=f"{name}-provider",
        provider_digest=_DIGEST,
        coordinate_authority_id="camera-native-v1",
        scale_authority_id="scale-v1",
        timing_authority_id="camera-time-v1",
        row_axis_authority_id="camera-rows-v1",
        row_axis_authority_digest=_DIGEST,
    )


def _record(schema_id: str, **values: object) -> dict[str, object]:
    return {"schema_id": schema_id, "schema_version": 1, **values}


def _prepared_base(*, include_optional: bool = True) -> tuple[dict[str, np.ndarray], ChaserRelativeDistanceRegistries]:
    n_frames = 3
    keys = AcquisitionFrameKeys(
        recording_id="recording-1",
        acquisition_frame_id=np.asarray([10, 11, 12], dtype=np.int64),
        track_sample_id=np.asarray([20, 21, 22], dtype=np.int64),
        row_axis_authority_id="camera-rows-v1",
        row_axis_authority_digest=_DIGEST,
        timestamp_ns=np.asarray([100, 200, 300], dtype=np.int64),
    )
    fish_xy = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    chaser_xy = np.asarray(
        [
            [[4.0, 1.0], [1.0, 5.0]],
            [[5.0, 2.0], [2.0, 6.0]],
            [[6.0, 3.0], [3.0, 7.0]],
        ]
    )
    roles = np.asarray(
        [["aggressive", "inert"], ["aggressive", "inert"], ["inert", "aggressive"]],
        dtype="<U16",
    )
    result = compute_chaser_relative_frame(
        ChaserRelativeFrameInput(
            frame_keys=keys,
            fish_xy=fish_xy,
            fish_valid=np.ones(n_frames, dtype=bool),
            fish_source_row_index=np.arange(n_frames, dtype=np.int64),
            fish_authority=_authority("fish"),
            chasers=ChaserObservations(
                identities=("chaser-a", "chaser-b"),
                behavior_roles=roles,
                xy=chaser_xy,
                valid=np.ones((n_frames, 2), dtype=bool),
                source_row_index=np.arange(6, dtype=np.int64).reshape(3, 2),
                authority=_authority("chaser"),
                trial_ids=(
                    np.asarray([[4, 4], [4, 4], [5, 5]], dtype=np.int64)
                    if include_optional
                    else None
                ),
                active=(
                    np.asarray([[True, True], [True, False], [True, True]])
                    if include_optional
                    else None
                ),
            ),
            selection_membership=np.asarray([True, True, False]),
            occurrence_membership=np.ones((n_frames, 2), dtype=bool),
            coordinate_policy=CoordinatePolicy(
                coordinate_authority_id="camera-native-v1",
                coordinate_frame="source_camera_pixels",
            ),
            scale_policy=ScalePolicy(
                scale_authority_id="scale-v1",
                scale_digest=_DIGEST,
                pixels_per_unit=10.0,
            ),
            timing_policy=TimingPolicy(
                timing_authority_id="camera-time-v1",
                timing_digest=_DIGEST,
                recording_id="recording-1",
            ),
        )
    )
    context = ChaserRelativeFramePublicationContext(
        fish_identity="fish-1",
        subject_identity_record=_record(
            "palette.subject_identity",
            recording_id="recording-1",
            subject_id="fish-1",
        ),
        temporal_selection_record=_record(
            "palette.temporal_selection",
            recording_id="recording-1",
            selection_id="all",
        ),
        chaser_occurrence_record=_record(
            "palette.chaser_occurrence",
            recording_id="recording-1",
            occurrence_policy_id="all",
        ),
        acquisition_projection_record=_record(
            "palette.acquisition_projection",
            recording_id="recording-1",
            policy_id="camera_frame_bound",
        ),
        analysis_profile_record=_record(
            "palette.analysis_profile",
            profile_id="chaser_behavior_full_v3",
        ),
    )
    prepared = prepare_chaser_relative_frame(result, context=context)
    base = {name: values.copy() for name, values in prepared.base_arrays.items()}
    registries = ChaserRelativeDistanceRegistries.from_manifest(
        prepared.manifest["identity_registries"], prepared.manifest["reason_codes"]
    )
    return base, registries


def _view_input(
    *,
    base: dict[str, np.ndarray] | None = None,
    registries: ChaserRelativeDistanceRegistries | None = None,
    path: str = "analysis/chaser_relative_frame_runs/run-1",
    digest: str = _DIGEST,
) -> ChaserRelativeDistanceViewInput:
    default_base, default_registries = _prepared_base()
    return ChaserRelativeDistanceViewInput(
        recording_id="recording-1",
        source_run_path=path,
        source_run_digest=digest,
        n_frames=3,
        n_chasers=2,
        base_arrays=default_base if base is None else base,
        registries=default_registries if registries is None else registries,
    )


def test_valid_view_reshapes_frame_and_pair_axes_and_nearest_one_hot() -> None:
    view = ChaserRelativeDistanceView.from_input(_view_input())

    assert view.n_rows == 6
    assert view.frame_array("acquisition_frame_id").shape == (3,)
    assert view.pair_array("relative_distance_px").shape == (3, 2)
    assert view.pair_array("relative_vector_physical_xy").shape == (3, 2, 2)
    nearest = view.pair_array("nearest_chaser_member")
    np.testing.assert_array_equal(
        nearest.sum(axis=1), view.frame_array("nearest_chaser_valid").astype(np.uint8)
    )
    assert view.frame_array("nearest_chaser_valid").any()
    assert view.chaser_identities == ("chaser-a", "chaser-b")


def test_repeated_frame_evidence_mismatch_fails_closed() -> None:
    base, _ = _prepared_base()
    base["timestamp_ns"][1] += 1
    with pytest.raises(ChaserRelativeDistanceViewError, match="timestamp_ns"):
        ChaserRelativeDistanceView.from_input(_view_input(base=base))


@pytest.mark.parametrize("field", ["acquisition_frame_id", "chaser_identity_code"])
def test_frame_order_or_identity_instability_fails_closed(field: str) -> None:
    base, _ = _prepared_base()
    if field == "acquisition_frame_id":
        base[field][1], base[field][2] = base[field][2], base[field][1]
    else:
        base[field][0] = 2
    with pytest.raises(ChaserRelativeDistanceViewError):
        ChaserRelativeDistanceView.from_input(_view_input(base=base))


def test_undeclared_behavior_code_fails_closed() -> None:
    base, _ = _prepared_base()
    base["chaser_behavior_role_code"][0] = 99
    with pytest.raises(ChaserRelativeDistanceViewError, match="behavior-role"):
        ChaserRelativeDistanceView.from_input(_view_input(base=base))


def test_schema_failure_rejects_missing_base_array() -> None:
    base, _ = _prepared_base()
    del base["relative_distance_px"]
    with pytest.raises(ChaserRelativeDistanceViewError, match="schema"):
        ChaserRelativeDistanceView.from_input(_view_input(base=base))


def test_optional_trial_and_active_arrays_are_preserved() -> None:
    base, _ = _prepared_base()
    view = ChaserRelativeDistanceView.from_input(_view_input(base=base))
    assert view.pair_array("trial_id").shape == (3, 2)
    assert view.pair_array("active_state_code").shape == (3, 2)

    base_without_optional, registries = _prepared_base(include_optional=False)
    reduced = ChaserRelativeDistanceView.from_input(
        _view_input(base=base_without_optional, registries=registries)
    )
    with pytest.raises(KeyError):
        reduced.pair_array("trial_id")


def test_view_arrays_are_c_contiguous_read_only_and_detached_from_input() -> None:
    base, _ = _prepared_base()
    view = ChaserRelativeDistanceView.from_input(_view_input(base=base))
    base["relative_distance_px"][0] = 999

    assert view.base_array("relative_distance_px")[0] != 999
    for array in (*view.base_arrays.values(), *view.frame_arrays.values(), *view.pair_arrays.values()):
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        view.pair_array("relative_distance_px")[0, 0] = 999


def test_position_only_view_has_no_body_dependency() -> None:
    base, registries = _prepared_base(include_optional=False)
    view = ChaserRelativeDistanceView.from_input(
        _view_input(base=base, registries=registries)
    )
    assert not hasattr(view, "body_arrays")
    assert "relative_vector_px_xy" in view.pair_arrays


def test_durable_view_path_requires_and_preserves_verified_source_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, registries = _prepared_base()
    verification_digest = "d" * 64
    fake = SimpleNamespace(
        recording_id="recording-1",
        run_path="analysis/chaser_relative_frame_runs/run-v1",
        verification_digest=verification_digest,
        n_frames=3,
        n_chasers=2,
        base_arrays=base,
        identity_registries={
            "fish": dict(registries.fish_identity),
            "chaser": dict(registries.chaser_identity),
            "behavior_role": dict(registries.behavior_role),
            "active_state": dict(registries.active_state or {}),
        },
        run_manifest={"reason_codes": dict(registries.reason)},
    )
    observed: list[object] = []

    def require(value: object):
        observed.append(value)
        return value

    monkeypatch.setattr(
        "fisheye.analysis_workflows.chaser_relative_distance_view."
        "require_chaser_relative_frame_source_handle",
        require,
    )

    view = load_chaser_relative_distance_view(fake)

    assert observed == [fake]
    assert view.source_run_digest == verification_digest
    assert view.chaser_identities == ("chaser-a", "chaser-b")
    assert view.pair_array("nearest_chaser_member").shape == (3, 2)


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        ("analysis/chaser_relative_frame_runs/latest", _DIGEST),
        ("analysis/chaser_relative_frame_runs/../run-1", _DIGEST),
        ("analysis/chaser_relative_frame_runs/run-1/child", _DIGEST),
        ("analysis/chaser_relative_frame_runs/run-1", "not-a-digest"),
    ],
)
def test_exact_source_identity_is_required(path: str, digest: str) -> None:
    with pytest.raises(ChaserRelativeDistanceViewError):
        ChaserRelativeDistanceView.from_input(_view_input(path=path, digest=digest))
