from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
)
from fisheye.analysis_workflows.controller_trial_successor import (
    controller_trial_input_from_handles,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorSourceHandle,
)
from fisheye.analysis_workflows.escape_freeze_successor import (
    escape_freeze_input_from_handles,
)
from fisheye.analysis_workflows.eye_gaze_source_handle import EyeGazeSourceHandle
from fisheye.analysis_workflows.gaze_tracking_successor import (
    gaze_tracking_input_from_handles,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    ROLE_CODES,
    generalized_bout_response_input_from_handles,
    prepare_generalized_bout_response_successor,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    ProtocolSemanticChaserSelectionSourceHandle,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    ProviderTrackMotionSourceHandle,
)
from tests.unit.fisheye.test_escape_freeze_successor import _dependencies


def _set(value: object, **fields: object) -> object:
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    return value


def _relative(archive: Path) -> ChaserRelativeFrameSourceHandle:
    frames = np.arange(100, 105, dtype=np.int64)
    base = {
        "acquisition_frame_id": frames,
        "timestamp_ns": np.arange(5, dtype=np.int64) * 1_000_000_000,
        "timestamp_valid": np.ones(5, dtype=bool),
        "chaser_identity_code": np.ones(5, dtype=np.uint16),
        "selection_member": np.zeros(5, dtype=bool),
        "chaser_occurrence_member": np.ones(5, dtype=bool),
        "fish_position_xy_px": np.tile([100.0, 100.0], (5, 1)).astype(np.float32),
        "fish_position_valid": np.ones(5, dtype=bool),
        "chaser_position_xy_px": np.asarray(
            [
                [110.0, 100.0],
                [110.0, 100.0],
                [110.0, 98.0],
                [108.0, 106.0],
                [109.0, 105.0],
            ],
            dtype=np.float32,
        ),
        "chaser_position_valid": np.ones(5, dtype=bool),
        "trial_id": np.ones(5, dtype=np.int64),
        "trial_valid": np.ones(5, dtype=bool),
        "active_state_code": np.ones(5, dtype=np.uint8),
        "active_state_valid": np.ones(5, dtype=bool),
        "relative_distance_physical": np.asarray(
            [10.0, 8.0, 12.0, 20.0, 30.0], dtype=np.float32
        ),
        "relative_physical_valid": np.ones(5, dtype=bool),
    }
    body = {
        "body_origin_xy_px": np.tile([100.0, 100.0], (5, 1)).astype(np.float32),
        "body_forward_axis_xy": np.tile([1.0, 0.0], (5, 1)).astype(np.float32),
        "body_left_axis_xy": np.tile([0.0, -1.0], (5, 1)).astype(np.float32),
        "body_axes_valid": np.ones(5, dtype=bool),
        "body_heading_deg": np.asarray([0.0, 0.0, 20.0, 20.0, 10.0], dtype=np.float32),
        "body_heading_valid": np.ones(5, dtype=bool),
        "body_bearing_deg": np.asarray(
            [30.0, 30.0, 10.0, -40.0, -30.0], dtype=np.float32
        ),
        "body_bearing_valid": np.ones(5, dtype=bool),
    }
    return _set(
        object.__new__(ChaserRelativeFrameSourceHandle),
        analysis_zarr_path=archive,
        run_path="analysis/chaser_relative_frame_runs/r1",
        run_name="r1",
        recording_id="recording-1",
        n_frames=5,
        n_chasers=1,
        n_rows=5,
        run_manifest={"scale_policy": {"unit": "mm", "pixels_per_unit": 10.0}},
        source_authorities={
            "fish_position": {
                "provider_id": "keypoint.v1",
                "provider_digest": "provider-digest",
            }
        },
        base_arrays=MappingProxyType(base),
        body_arrays=MappingProxyType(body),
        context={
            "temporal_selection": {
                "record": {"selection_id": "exact-selection"},
                "sha256": "f" * 64,
            }
        },
    )


def _semantic(archive: Path) -> ProtocolSemanticChaserSelectionSourceHandle:
    bounds = {
        "chaser_pre": (90, 95),
        "chaser_training": (100, 105),
        "chaser_post": (110, 115),
    }
    return _set(
        object.__new__(ProtocolSemanticChaserSelectionSourceHandle),
        analysis_zarr=archive,
        recording_id="recording-1",
        run_name="s1",
        run_path="analysis/protocol_semantic_chaser_selection_runs/s1",
        manifest={
            "role_records": [
                {
                    "role": role,
                    "selected_start_frame": bounds[role][0],
                    "selected_end_frame_exclusive": bounds[role][1],
                }
                for role in ROLE_CODES
            ]
        },
    )


def _provider(archive: Path) -> ProviderTrackMotionSourceHandle:
    frames = np.arange(100, 105, dtype=np.int64)
    return _set(
        object.__new__(ProviderTrackMotionSourceHandle),
        analysis_zarr_path=archive,
        run_path="analysis/track_kinematics_runs/provider/m1",
        run_name="m1",
        provider_manifest_sha256="b" * 64,
        verification_digest="9" * 64,
        arrays=MappingProxyType(
            {
                "track_ids": np.asarray([7], dtype=np.int64),
                "track_row_offsets": np.asarray([0, 5], dtype=np.int64),
                "source_acquisition_frame_index": frames,
                "transition_valid": np.asarray(
                    [False, True, True, True, True], dtype=bool
                ),
                "linear_sample_valid": np.ones(5, dtype=bool),
                "speed_filtered_mm": np.asarray(
                    [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                ),
            }
        ),
    )


def _eye(archive: Path) -> EyeGazeSourceHandle:
    gaze = np.full((105, 2), np.nan, dtype=np.float64)
    gaze[100:105] = np.asarray([[1.0, 2.0]] * 5)
    vergence = np.full(105, np.nan, dtype=np.float64)
    vergence[100:105] = 3.0
    valid = np.isfinite(gaze)
    return _set(
        object.__new__(EyeGazeSourceHandle),
        analysis_zarr_path=archive,
        run_path="analysis/eye_angle_runs/eye-v7",
        run_name="eye-v7",
        recording_id="recording-1",
        n_frames=105,
        logical_manifest_sha256="e" * 64,
        convention_receipt_sha256="f" * 64,
        channel_variant="smoothed",
        gaze_channel_names=(
            "left_gaze_signed_deg_smoothed",
            "right_gaze_signed_deg_smoothed",
        ),
        vergence_channel_name="vergence_eye_angle_deg_smoothed",
        gaze_signed_deg=gaze,
        gaze_valid=valid,
        vergence_deg=vergence,
        vergence_valid=np.isfinite(vergence),
    )


def _radial(archive: Path) -> ComposableChaserSuccessorSourceHandle:
    scientific = {
        "sources": {
            "relative_frame": {
                "run_path": "analysis/chaser_relative_frame_runs/r1",
                "manifest_sha256": "a" * 64,
            },
            "protocol_semantic_selection": {
                "run_path": "analysis/protocol_semantic_chaser_selection_runs/s1",
                "manifest_sha256": "a" * 64,
            },
            "arena_geometry_and_scale": {"authority_sha256": "8" * 64},
        },
        "position_provider": {
            "provider_id": "keypoint.v1",
            "provider_digest": "provider-digest",
            "status": "first_class_explicit_authority",
        },
        "arena": {
            "center_xy_px": [100.0, 100.0],
            "radius_px": 200.0,
            "radius_mm": 20.0,
        },
    }
    return _set(
        object.__new__(ComposableChaserSuccessorSourceHandle),
        analysis_zarr=archive,
        successor_kind="chaser_radial_near_field",
        run_path="analysis/chaser_radial_near_field_runs/radial-v1",
        run_name="radial-v1",
        recording_id="recording-1",
        manifest={
            "scientific_manifest": scientific,
            "scientific_payload_sha256": "9" * 64,
        },
    )


def _install_current_handle_stubs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle,
        "manifest_sha256",
        property(lambda self: "a" * 64),
    )
    monkeypatch.setattr(
        ProtocolSemanticChaserSelectionSourceHandle,
        "assert_current",
        lambda self: None,
    )
    monkeypatch.setattr(
        ProtocolSemanticChaserSelectionSourceHandle,
        "manifest_sha256",
        property(lambda self: "a" * 64),
    )
    monkeypatch.setattr(
        ProtocolSemanticChaserSelectionSourceHandle,
        "source_binding",
        lambda self: {"source_epoch_selection": {"selection_id": "exact-selection"}},
    )
    monkeypatch.setattr(
        ProviderTrackMotionSourceHandle, "assert_current", lambda self: None
    )
    monkeypatch.setattr(EyeGazeSourceHandle, "assert_current", lambda self: None)
    monkeypatch.setattr(
        ComposableChaserSuccessorSourceHandle, "assert_current", lambda self: None
    )


def test_gaze_adapter_binds_reviewed_channels_and_semantic_axis(
    tmp_path, monkeypatch
) -> None:
    _install_current_handle_stubs(monkeypatch)
    archive = (tmp_path / "analysis.zarr").resolve()

    source = gaze_tracking_input_from_handles(
        _relative(archive), _semantic(archive), _eye(archive), _radial(archive)
    )

    assert source.acquisition_frame_id_by_frame.tolist() == [100, 101, 102, 103, 104]
    assert source.semantic_role_code_by_frame.tolist() == [2, 2, 2, 2, 2]
    assert source.source_eye_channel_policy.startswith("smoothed:")
    np.testing.assert_allclose(source.gaze_signed_deg, [[1.0, 2.0]] * 5)
    assert source.source_radial_run_path.endswith("/radial-v1")


def test_controller_adapter_projects_semantic_bounds_by_acquisition_frame(
    tmp_path, monkeypatch
) -> None:
    _install_current_handle_stubs(monkeypatch)
    archive = (tmp_path / "analysis.zarr").resolve()

    source = controller_trial_input_from_handles(_relative(archive), _semantic(archive))

    # The stored relative-frame selection is deliberately all false and its
    # temporal context is not the epoch-selection record. Exact semantic
    # membership comes from joining the published role bounds to frame IDs.
    assert source.selection_member.tolist() == [True, True, True, True, True]
    assert source.acquisition_frame_id.tolist() == [100, 101, 102, 103, 104]


def test_generalized_and_escape_adapters_share_exact_axes_and_dependencies(
    tmp_path, monkeypatch
) -> None:
    _install_current_handle_stubs(monkeypatch)
    archive = (tmp_path / "analysis.zarr").resolve()
    relative = _relative(archive)
    semantic = _semantic(archive)
    provider = _provider(archive)
    controller, _unused_bout = _dependencies()
    bouts = np.zeros(
        1,
        dtype=np.dtype(
            [
                ("bout_id", "i4"),
                ("start_frame", "i8"),
                ("end_frame", "i8"),
                ("peak_physical_speed_mm_s", "f8"),
                ("mean_speed_mm_s", "f8"),
                ("duration_s", "f8"),
                ("path_length_mm", "f8"),
                ("net_displacement_mm", "f8"),
            ]
        ),
    )
    bouts[0] = (41, 101, 102, 25.0, 12.0, 0.2, 3.0, 2.0)
    tables = SimpleNamespace(
        bouts=bouts,
        run_path="analysis/swim_bout_runs/b1",
        signal=SimpleNamespace(signal_id=4, speed_level="speed_filtered"),
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr_io.open_zarr_root", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "fisheye.analysis.swim_bout_io.load_exact_selector_ineligible_default_swim_bout_tables",
        lambda *args, **kwargs: tables,
    )
    monkeypatch.setattr(
        "fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary._swim_bout_binding",
        lambda *args, **kwargs: (
            {"run_path": "analysis/swim_bout_runs/b1"},
            "c" * 64,
            "d" * 64,
        ),
    )

    generalized_input = generalized_bout_response_input_from_handles(
        relative,
        semantic,
        controller,
        provider,
        swim_bout_run_name="b1",
        track_id=7,
    )
    generalized = prepare_generalized_bout_response_successor(generalized_input)
    escape_input = escape_freeze_input_from_handles(
        relative,
        provider,
        controller,
        generalized,
        track_id=7,
        speed_level="filtered",
    )

    assert generalized_input.source_signal_id == 4
    assert generalized.array("controller_trial_row_id").tolist() == [0]
    assert escape_input.source_speed_level == "filtered"
    np.testing.assert_allclose(escape_input.speed_mm_s_by_frame, [1, 2, 3, 4, 5])
    assert escape_input.acquisition_frame_id_by_frame.tolist() == [
        100,
        101,
        102,
        103,
        104,
    ]
