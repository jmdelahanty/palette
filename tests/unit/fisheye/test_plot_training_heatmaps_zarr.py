from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

import fisheye.analysis.plot_training_heatmaps_zarr as heatmaps


def _track(
    track_id: int,
    *,
    extent: tuple[int, int] = (640, 480),
    manifest_sha256: str = "a" * 64,
    direct: bool = True,
) -> SimpleNamespace:
    frames = np.asarray([2 * track_id, 2 * track_id + 1], dtype=np.int64)
    positions = np.asarray(
        [[10.0 + track_id, 20.0], [30.0 + track_id, 40.0]],
        dtype=np.float64,
    )

    def _require() -> tuple[np.ndarray, int, int]:
        if not direct:
            raise ValueError("positions_px is not directly suitable")
        return positions, extent[0], extent[1]

    return SimpleNamespace(
        track_id=track_id,
        track_path=(
            "analysis/track_kinematics_runs/offline/tk/tracks/"
            f"id_{track_id}"
        ),
        source_acquisition_frame_index=frames,
        sample_valid=np.asarray([True, track_id == 0], dtype=bool),
        motion_manifest_sha256=manifest_sha256,
        require_direct_source_camera_positions_px=_require,
        authority_record=lambda: {
            "schema_id": "palette.track_motion_read_authority",
            "track_id": track_id,
            "motion_manifest_sha256": manifest_sha256,
        },
    )


def _install_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tracks: dict[int, SimpleNamespace],
) -> list[dict[str, object]]:
    run_group = object()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        heatmaps,
        "resolve_track_kinematics_run",
        lambda _root, **kwargs: (
            run_group,
            "tk",
            "analysis/track_kinematics_runs/offline/tk",
        ),
    )
    monkeypatch.setattr(
        heatmaps,
        "list_track_ids",
        lambda value: sorted(tracks) if value is run_group else [],
    )

    def _load(_root: object, **kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return tracks[int(kwargs["track_id"])]

    monkeypatch.setattr(heatmaps, "load_track_kinematics_track", _load)
    return calls


def test_collect_positions_uses_verified_camera_extent_and_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_catalog(monkeypatch, {0: _track(0), 1: _track(1)})

    result = heatmaps._collect_positions(
        object(),
        None,
        Console(file=StringIO()),
    )

    assert result.run_label == "offline/tk"
    assert (result.reference_width, result.reference_height) == (640, 480)
    assert result.frames.tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        result.positions,
        [[10.0, 20.0], [30.0, 40.0], [11.0, 20.0]],
    )
    assert [call["required_speed_levels"] for call in calls] == [(), ()]


def test_collect_positions_rejects_transform_required_texture_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_catalog(monkeypatch, {0: _track(0, direct=False)})

    with pytest.raises(ValueError, match="not directly suitable"):
        heatmaps._collect_positions(
            object(),
            "offline/tk",
            Console(file=StringIO()),
        )


def test_collect_positions_rejects_reference_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_catalog(
        monkeypatch,
        {0: _track(0), 1: _track(1, extent=(800, 600))},
    )

    with pytest.raises(ValueError, match="reference dimensions"):
        heatmaps._collect_positions(
            object(),
            "offline/tk",
            Console(file=StringIO()),
        )


def test_collect_positions_rejects_manifest_change_between_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_catalog(
        monkeypatch,
        {0: _track(0), 1: _track(1, manifest_sha256="b" * 64)},
    )

    with pytest.raises(ValueError, match="authority changed"):
        heatmaps._collect_positions(
            object(),
            "offline/tk",
            Console(file=StringIO()),
        )


def test_track_run_spec_is_explicit_and_defaults_to_offline() -> None:
    assert heatmaps._track_run_spec(None) == ("offline", "latest")
    assert heatmaps._track_run_spec("online/run") == ("online", "run")
    assert heatmaps._track_run_spec(
        "analysis/track_kinematics_runs/offline/run"
    ) == ("offline", "run")
    with pytest.raises(ValueError, match="canonical analysis path"):
        heatmaps._track_run_spec("online/run/extra")
