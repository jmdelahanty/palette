from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

import fisheye.analysis.measure_noise_floor as noise_floor


def _track(
    *,
    frames: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    sample_valid: np.ndarray | None = None,
    transition_valid: np.ndarray | None = None,
    manifest_sha256: str = "a" * 64,
) -> SimpleNamespace:
    frame_values = np.asarray(
        [0, 1, 2, 10, 11, 12] if frames is None else frames,
        dtype=np.int64,
    )
    row_count = frame_values.shape[0]
    return SimpleNamespace(
        track_path="analysis/track_kinematics_runs/offline/tk/tracks/id_0",
        source_acquisition_frame_index=frame_values,
        sample_valid=np.ones(row_count, dtype=bool)
        if sample_valid is None
        else sample_valid,
        transition_valid=np.asarray(
            [False, True, True, True, True, True]
            if transition_valid is None
            else transition_valid,
            dtype=bool,
        ),
        speed_mm_by_level={
            "raw": np.asarray(
                [np.nan, 0.1, 0.2, 0.1, 0.2, 0.1]
                if speed is None
                else speed,
                dtype=np.float64,
            )
        },
        frame_path_distance_mm_by_level={
            "raw": np.asarray(
                [np.nan, 0.01, 0.02, 0.01, 0.02, 0.01]
                if distance is None
                else distance,
                dtype=np.float64,
            )
        },
        motion_manifest_sha256=manifest_sha256,
        run_attrs={
            "method": "verified",
            "fps": 10.0,
            "physical_outputs_available": True,
        },
    )


def test_identify_stationary_periods_does_not_bridge_acquisition_gap() -> None:
    periods = noise_floor.identify_stationary_periods(
        np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
        np.asarray([0, 1, 10, 11], dtype=np.int64),
        min_period_frames=2,
    )

    assert periods == [(0, 2), (2, 4)]


def test_compute_noise_statistics_uses_verified_raw_maps_and_validity() -> None:
    statistics = noise_floor.compute_noise_statistics(
        {"id_0": _track()},
        max_stationary_speed=0.5,
        min_period_frames=2,
        console=Console(file=StringIO()),
    )

    assert statistics["n_tracks"] == 1
    assert statistics["total_stationary_frames"] == 5
    assert statistics["speed_mean"] == pytest.approx(0.14)
    assert statistics["path_distance_mean"] == pytest.approx(0.014)


def test_load_track_kinematics_run_uses_strict_reader_for_every_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = object()
    run_group = object()
    calls: list[int] = []
    monkeypatch.setattr(noise_floor, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        noise_floor,
        "resolve_track_kinematics_run",
        lambda *_args, **_kwargs: (run_group, "tk", "analysis/track_kinematics_runs/offline/tk"),
    )
    monkeypatch.setattr(noise_floor, "list_track_ids", lambda value: [0, 2] if value is run_group else [])

    def _load(_root: object, **kwargs: object) -> SimpleNamespace:
        calls.append(int(kwargs["track_id"]))
        value = _track()
        value.track_path = (
            "analysis/track_kinematics_runs/offline/tk/tracks/"
            f"id_{int(kwargs['track_id'])}"
        )
        return value

    monkeypatch.setattr(noise_floor, "load_track_kinematics_track", _load)

    attrs, tracks = noise_floor.load_track_kinematics_run(
        "/archive.zarr",
        console=Console(file=StringIO()),
    )

    assert calls == [0, 2]
    assert set(tracks) == {"id_0", "id_2"}
    assert attrs["physical_outputs_available"] is True


def test_load_track_kinematics_run_rejects_authority_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(noise_floor, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        noise_floor,
        "resolve_track_kinematics_run",
        lambda *_args, **_kwargs: (object(), "tk", "analysis/track_kinematics_runs/offline/tk"),
    )
    monkeypatch.setattr(noise_floor, "list_track_ids", lambda _value: [0, 1])
    monkeypatch.setattr(
        noise_floor,
        "load_track_kinematics_track",
        lambda _root, **kwargs: _track(
            manifest_sha256=("a" if int(kwargs["track_id"]) == 0 else "b") * 64
        ),
    )

    with pytest.raises(ValueError, match="authority changed"):
        noise_floor.load_track_kinematics_run(
            "/archive.zarr",
            console=Console(file=StringIO()),
        )
