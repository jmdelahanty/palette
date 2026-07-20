from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fisheye.analysis import inspect_heatmap_inputs_zarr as mod


def test_inspector_uses_verified_track_motion_reader(monkeypatch, tmp_path) -> None:
    stimulus_run = object()
    root = {
        "analysis": {
            "stimulus_runs": {"stim_1": stimulus_run},
        }
    }
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        mod,
        "_resolve_latest",
        lambda parent, requested, _label, _console: (stimulus_run, "stim_1"),
    )
    monkeypatch.setattr(mod, "_load_events", lambda _group: {})
    monkeypatch.setattr(
        mod,
        "_load_frame_metadata",
        lambda _group: {
            "triggering_camera_frame_id": np.asarray([10, 11], dtype=np.int64)
        },
    )
    monkeypatch.setattr(mod, "_build_stim_to_camera_map", lambda _metadata: {})
    monkeypatch.setattr(
        mod,
        "_determine_periods",
        lambda *_args: SimpleNamespace(
            pre_start=10,
            pre_end=10,
            train_start=11,
            train_end=11,
            post_start=12,
            post_end=12,
        ),
    )
    calls = []

    def collect(root_arg, run_spec, console):
        calls.append((root_arg, run_spec, console))
        return SimpleNamespace(
            run_label="offline/motion_1",
            frames=np.asarray([10, 11], dtype=np.int64),
            positions=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        )

    monkeypatch.setattr(mod, "_collect_positions", collect)

    mod.analyze_heatmap_inputs(
        tmp_path / "analysis.zarr",
        "offline/motion_1",
        "stim_1",
    )

    assert len(calls) == 1
    assert calls[0][0] is root
    assert calls[0][1] == "offline/motion_1"
