from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis.megabouts_preprocessing_comparison import (
    MegaboutsPreprocessingRuntime,
    _array_stats,
    build_megabouts_preprocessed_input_pack,
    compare_megabouts_preprocessing_with_palette,
)
from tests.unit.fisheye.test_megabouts_classifier_inputs import (
    _build_root,
    _install_verified_source_readers,
    _reason_bytes,
    _replace_array,
    build_megabouts_classifier_input_pack,
)


@pytest.fixture(autouse=True)
def _verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_source_readers(monkeypatch)


class _FakeTailPreprocessingConfig:
    def __init__(self, *, fps: int) -> None:
        self.fps = int(fps)


class _FakeTrajPreprocessingConfig:
    def __init__(self, *, fps: int) -> None:
        self.fps = int(fps)


class _FakeTailPreprocessing:
    def __init__(self, config: _FakeTailPreprocessingConfig) -> None:
        self.config = config

    def preprocess_tail_df(self, tail_df):
        angle = tail_df[[f"angle_{idx}" for idx in range(10)]].to_numpy(dtype=np.float32)
        no_tracking = np.any(~np.isfinite(angle), axis=1)
        angle_smooth = np.where(np.isfinite(angle), angle + 0.5, 0.0).astype(np.float32)
        return SimpleNamespace(
            angle=angle,
            angle_baseline=np.zeros_like(angle_smooth),
            angle_smooth=angle_smooth,
            vigor=np.zeros((angle.shape[0],), dtype=np.float32),
            no_tracking=no_tracking,
        )


class _FakeTrajPreprocessing:
    def __init__(self, config: _FakeTrajPreprocessingConfig) -> None:
        self.config = config

    def preprocess_traj_df(self, traj_df):
        x = traj_df["x"].to_numpy(dtype=np.float32)
        y = traj_df["y"].to_numpy(dtype=np.float32)
        yaw = traj_df["yaw"].to_numpy(dtype=np.float32)
        no_tracking = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(yaw)
        return SimpleNamespace(
            x=x,
            y=y,
            yaw=yaw,
            x_smooth=np.where(np.isfinite(x), x + 10.0, 0.0).astype(np.float32),
            y_smooth=np.where(np.isfinite(y), y + 20.0, 0.0).astype(np.float32),
            yaw_smooth=np.where(np.isfinite(yaw), yaw + 0.25, 0.0).astype(np.float32),
            axial_speed=np.zeros((x.shape[0],), dtype=np.float32),
            lateral_speed=np.zeros((x.shape[0],), dtype=np.float32),
            yaw_speed=np.zeros((x.shape[0],), dtype=np.float32),
            vigor=np.zeros((x.shape[0],), dtype=np.float32),
            no_tracking=no_tracking,
        )


def _fake_preprocessing_runtime() -> MegaboutsPreprocessingRuntime:
    return MegaboutsPreprocessingRuntime(
        tail_preprocessing_class=_FakeTailPreprocessing,
        tail_preprocessing_config_class=_FakeTailPreprocessingConfig,
        traj_preprocessing_class=_FakeTrajPreprocessing,
        traj_preprocessing_config_class=_FakeTrajPreprocessingConfig,
        package_version="fake-0",
        package_path="/fake/megabouts",
        source_repo="/fake/megabouts_repo",
        git_commit="fakecommit",
    )


def _prepend_competing_posture_instance(root) -> np.ndarray:
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    original_angles = np.asarray(posture["tail_angle_rad"][:])
    original_valid = np.asarray(posture["valid"][:])
    original_reasons = np.asarray(posture["failure_reason_bytes"][:])
    _replace_array(
        posture,
        "source_acquisition_frame_index",
        np.asarray([1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
    )
    _replace_array(
        posture,
        "instance_key",
        np.asarray([999, 100, 101, 102, 103, 104, 105, 106, 107], dtype=np.uint64),
    )
    _replace_array(
        posture,
        "tail_angle_rad",
        np.concatenate(
            [np.full((1, 10), 99.0, dtype=np.float32), original_angles],
            axis=0,
        ),
    )
    _replace_array(
        posture,
        "valid",
        np.concatenate([np.asarray([True]), original_valid]),
    )
    _replace_array(
        posture,
        "failure_reason_bytes",
        np.concatenate([_reason_bytes(["wrong_animal"]), original_reasons], axis=0),
    )
    return original_angles


def test_build_megabouts_preprocessed_input_pack_uses_same_windows() -> None:
    root = _build_root()
    source_pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.75,
        max_consecutive_invalid_frames=1,
    )

    preprocessed = build_megabouts_preprocessed_input_pack(
        root,
        source_pack=source_pack,
        runtime=_fake_preprocessing_runtime(),
    )

    assert preprocessed.window_start_frame.tolist() == source_pack.window_start_frame.tolist()
    assert preprocessed.window_end_frame.tolist() == source_pack.window_end_frame.tolist()
    assert preprocessed.tail_valid.tolist() == source_pack.tail_valid.tolist()
    assert preprocessed.traj_valid.tolist() == source_pack.traj_valid.tolist()
    assert preprocessed.valid_bout.tolist() == source_pack.valid_bout.tolist()
    assert preprocessed.parameters["classifier_input_mode"] == "megabouts_preprocessed_full_timeseries"
    assert preprocessed.parameters["calls_megabouts_preprocessing"] is True
    assert preprocessed.parameters["megabouts_preprocessing_config"]["tail"]["parameters"]["fps"] == 10
    assert preprocessed.parameters["megabouts_preprocessing_config"]["trajectory"]["parameters"]["fps"] == 10
    np.testing.assert_allclose(
        preprocessed.tail_array[0, :, 0],
        source_pack.tail_array[0, :, 0] + 0.5,
        rtol=1e-6,
        atol=1e-6,
    )


def test_preprocessed_tail_uses_track_source_instance_key_on_shared_frame() -> None:
    root = _build_root()
    original_angles = _prepend_competing_posture_instance(root)
    source_pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=1,
        min_tail_valid_fraction=0.0,
        min_traj_valid_fraction=0.0,
    )

    preprocessed = build_megabouts_preprocessed_input_pack(
        root,
        source_pack=source_pack,
        runtime=_fake_preprocessing_runtime(),
    )

    np.testing.assert_allclose(
        preprocessed.tail_array[0, :, 0],
        original_angles[1] + 0.5,
    )
    assert not np.all(preprocessed.tail_array[0, :, 0] == 99.5)


def test_preprocessed_tail_rejects_instance_key_frame_disagreement() -> None:
    root = _build_root()
    source_pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=1,
    )
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    frames = np.asarray(posture["source_acquisition_frame_index"][:])
    frames[1] = 2
    _replace_array(posture, "source_acquisition_frame_index", frames)

    with pytest.raises(ValueError, match="acquisition-frame mismatch"):
        build_megabouts_preprocessed_input_pack(
            root,
            source_pack=source_pack,
            runtime=_fake_preprocessing_runtime(),
        )


def test_compare_megabouts_preprocessing_reports_input_similarity() -> None:
    root = _build_root()

    report = compare_megabouts_preprocessing_with_palette(
        root,
        runtime=_fake_preprocessing_runtime(),
        swim_bout_legacy_compatibility=True,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.75,
        max_consecutive_invalid_frames=1,
    )

    assert report["status"] == "ok"
    assert report["mutates_archive"] is False
    assert report["calls_megabouts_preprocessing"] is True
    assert report["calls_megabouts_classifier"] is False
    assert report["source_bout_count"] == 2
    assert report["common_valid_bout_count"] == 2
    assert report["tail_angle_comparison_rad"]["overall"]["n"] == 70
    assert np.isclose(report["tail_angle_comparison_rad"]["overall"]["mean_abs"], 0.5)
    assert report["tail_validity_disagreement_count"] == 0
    assert report["traj_validity_disagreement_count"] == 0


def test_array_stats_can_compare_wrapped_radian_angles() -> None:
    a = np.asarray([np.pi - 0.1], dtype=np.float32)
    b = np.asarray([-np.pi + 0.1], dtype=np.float32)
    mask = np.asarray([True], dtype=bool)

    linear = _array_stats(a, b, mask)
    angular = _array_stats(a, b, mask, angular_radians=True)

    assert linear["max_abs"] > 6.0
    assert np.isclose(angular["max_abs"], 0.2, atol=1e-6)
