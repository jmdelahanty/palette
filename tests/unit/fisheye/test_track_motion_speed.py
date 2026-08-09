from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis.track_motion_speed as speed_module
from fisheye.analysis.track_motion_speed import load_verified_smoothed_frame_speed


def _track(
    *,
    frames: list[int],
    speed: list[float],
    sample_valid: list[bool],
    transition_valid: list[bool],
) -> SimpleNamespace:
    return SimpleNamespace(
        track_path="analysis/track_kinematics_runs/offline/tk/tracks/id_0",
        source_acquisition_frame_index=np.asarray(frames, dtype=np.int64),
        speed_mm_by_level={"smoothed": np.asarray(speed, dtype=np.float64)},
        sample_valid=np.asarray(sample_valid, dtype=bool),
        transition_valid=np.asarray(transition_valid, dtype=bool),
        authority_record=lambda: {
            "schema_id": "palette.track_motion_read_authority",
            "schema_version": 1,
            "track_ref": "analysis/track_kinematics_runs/offline/tk/tracks/id_0",
        },
    )


def test_verified_speed_excludes_invalid_transitions_across_tracking_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = _track(
        frames=[0, 1, 2, 4],
        speed=[0.0, 1.0, 999.0, 2.0],
        sample_valid=[True, True, True, True],
        transition_valid=[False, True, False, True],
    )
    monkeypatch.setattr(
        speed_module,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: track,
    )

    result = load_verified_smoothed_frame_speed({}, 5)

    np.testing.assert_allclose(
        result.values_mm_s,
        [np.nan, 1.0, np.nan, np.nan, 2.0],
        equal_nan=True,
    )
    assert result.source == "track_motion.movement/speed/smoothed/mm"


def test_verified_speed_rejects_duplicate_frame_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = _track(
        frames=[0, 1, 1],
        speed=[0.0, 1.0, 2.0],
        sample_valid=[True, True, True],
        transition_valid=[False, True, True],
    )
    monkeypatch.setattr(
        speed_module,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: track,
    )

    with pytest.raises(ValueError, match="repeats acquisition-frame identities"):
        load_verified_smoothed_frame_speed({}, 3)
