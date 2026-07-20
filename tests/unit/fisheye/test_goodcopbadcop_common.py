from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis.goodcopbadcop_common as common


def _verified_track(**overrides: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "track_path": "analysis/track_kinematics_runs/offline/tk/tracks/id_0",
        "source_acquisition_frame_index": np.asarray([1, 3, 4], dtype=np.int64),
        "sample_valid": np.asarray([True, False, True], dtype=bool),
        "speed_mm_by_level": {
            "raw": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "smoothed": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        },
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_load_dense_kinematics_uses_verified_acquisition_frame_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _load(_root: object, **kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return _verified_track()

    monkeypatch.setattr(common, "load_track_kinematics_track", _load)

    arrays, valid = common.load_dense_kinematics(
        object(),
        6,
        fields=("speed_smoothed_mm", "speed_raw_mm"),
        track_kinematics_run="sealed_run",
    )

    np.testing.assert_allclose(
        arrays["speed_smoothed_mm"],
        [np.nan, 0.1, np.nan, 0.2, 0.3, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        arrays["speed_raw_mm"],
        [np.nan, 1.0, np.nan, 2.0, 3.0, np.nan],
        equal_nan=True,
    )
    assert valid.tolist() == [False, True, False, False, True, False]
    assert calls == [
        {
            "run_name": "sealed_run",
            "scope": "offline",
            "track_id": 0,
            "required_speed_levels": ("smoothed", "raw"),
        }
    ]


def test_load_dense_kinematics_rejects_uncontrolled_field_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: pytest.fail("strict reader should not be called"),
    )

    with pytest.raises(ValueError, match="Unsupported verified dense-track field"):
        common.load_dense_kinematics(object(), 6, fields=("positions_mm",))


def test_load_dense_kinematics_rejects_out_of_extent_frame_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: _verified_track(
            source_acquisition_frame_index=np.asarray([1, 3, 6], dtype=np.int64)
        ),
    )

    with pytest.raises(ValueError, match="declared recording extent"):
        common.load_dense_kinematics(object(), 6)


def test_load_dense_kinematics_rejects_missing_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        common,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: _verified_track(sample_valid=None),
    )

    with pytest.raises(ValueError, match="no sample_valid"):
        common.load_dense_kinematics(object(), 6)
