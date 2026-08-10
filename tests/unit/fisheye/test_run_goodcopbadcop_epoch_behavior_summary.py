from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.utils import run_goodcopbadcop_epoch_behavior_summary as mod
from fisheye.utils import run_chaser_epoch_behavior_summary as generic_mod
from fisheye.analysis.chaser_epoch_behavior_summary import (
    AUTHORITATIVE_EXECUTION_MODE,
    LEGACY_EXECUTION_MODE,
)


def test_run_for_targets_forwards_epoch_behavior_parameters_and_writes(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    zarr_path = tmp_path / "recording_GoodCopBadCop_analysis.zarr"
    zarr_path.mkdir()

    def fake_build(path: Path, **kwargs):
        calls["build_path"] = path
        calls["build_kwargs"] = kwargs
        return SimpleNamespace(
            chaser_distance_run_name="chaser_distance_1",
            source_swim_bout_run="bouts_1",
            source_track_kinematics_run="tk_1",
            per_epoch_fish=np.asarray(
                [
                    (0, 2, 1),
                    (1, 1, 0),
                ],
                dtype=[
                    ("window_id", "i4"),
                    ("bout_count", "i8"),
                    ("inter_bout_interval_count", "i8"),
                ],
            ),
            per_epoch_chaser=np.zeros(4, dtype=[("window_id", "i4")]),
            per_epoch_bouts=np.zeros(3, dtype=[("window_id", "i4")]),
            per_epoch_bout_histograms=np.zeros(12, dtype=[("window_id", "i4")]),
            per_epoch_inter_bout_interval_histograms=np.zeros(6, dtype=[("window_id", "i4")]),
            center_distance_histogram=np.zeros(6, dtype=[("window_id", "i4")]),
            warnings=("track_kinematics_unavailable: test",),
        )

    def fake_write(path: Path, result, **kwargs):
        calls["write_path"] = path
        calls["write_result"] = result
        calls["write_kwargs"] = kwargs
        return "analysis/chaser_distance_runs/chaser_distance_1/epoch_behavior_summary/kinematics_bouts_v1"

    monkeypatch.setattr(generic_mod, "build_chaser_epoch_behavior_summary_result", fake_build)
    monkeypatch.setattr(generic_mod, "write_chaser_epoch_behavior_summary_component", fake_write)

    rows = mod.run_for_targets(
        [
            {
                "recording_id": "recording_GoodCopBadCop",
                "zarr_path": str(zarr_path),
                "coverage_percent": 99.0,
                "detect_run": "detect_1",
                "refined_run": "refined_1",
            }
        ],
        chaser_distance_run="latest",
        component_name="kinematics_bouts_v1",
        swim_bout_run="latest",
        track_kinematics_run="tk_1",
        track_kinematics_scope="offline",
        track_id=0,
        speed_level="filtered",
        apply=True,
        overwrite=True,
    )

    assert calls["build_path"] == zarr_path
    assert calls["build_kwargs"] == {
        "chaser_distance_run": "latest",
        "component_name": "kinematics_bouts_v1",
        "swim_bout_run": "latest",
        "track_kinematics_run": "tk_1",
        "track_kinematics_scope": "offline",
        "track_id": 0,
        "speed_level": "filtered",
        "execution_mode": AUTHORITATIVE_EXECUTION_MODE,
        "center_distance_bin_width_mm": mod.DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM,
        "wall_band_mm": mod.DEFAULT_WALL_BAND_MM,
    }
    assert calls["write_path"] == zarr_path
    assert calls["write_kwargs"] == {"overwrite": True}
    assert rows == [
        {
            "recording_id": "recording_GoodCopBadCop",
            "zarr_path": str(zarr_path),
            "detect_coverage_percent": 99.0,
            "detect_run": "detect_1",
            "refined_run": "refined_1",
            "chaser_distance_run": "chaser_distance_1",
            "source_swim_bout_run": "bouts_1",
            "source_track_kinematics_run": "tk_1",
            "component_name": "kinematics_bouts_v1",
            "execution_mode": AUTHORITATIVE_EXECUTION_MODE,
            "epoch_behavior_summary_path": "analysis/chaser_distance_runs/chaser_distance_1/epoch_behavior_summary/kinematics_bouts_v1",
            "status": "complete",
            "error": None,
            "summary": {
                "epoch_count": 2,
                "chaser_epoch_count": 4,
                "per_epoch_bout_count": 3,
                "per_epoch_bout_histogram_count": 12,
                "per_epoch_inter_bout_interval_histogram_count": 6,
                "center_distance_histogram_count": 6,
                "bout_count": [2, 1],
                "inter_bout_interval_count": [1, 0],
                "warnings": ["track_kinematics_unavailable: test"],
            },
        }
    ]


def test_run_for_targets_dry_run_does_not_write(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_GoodCopBadCop_analysis.zarr"
    zarr_path.mkdir()
    writes: list[object] = []

    def fake_build(path: Path, **kwargs):
        return SimpleNamespace(
            chaser_distance_run_name="chaser_distance_1",
            source_swim_bout_run=None,
            source_track_kinematics_run=None,
            per_epoch_fish=np.zeros(
                1,
                dtype=[
                    ("bout_count", "i8"),
                    ("inter_bout_interval_count", "i8"),
                ],
            ),
            per_epoch_chaser=np.zeros(0, dtype=[("window_id", "i4")]),
            per_epoch_bouts=np.zeros(0, dtype=[("window_id", "i4")]),
            per_epoch_bout_histograms=np.zeros(0, dtype=[("window_id", "i4")]),
            per_epoch_inter_bout_interval_histograms=np.zeros(0, dtype=[("window_id", "i4")]),
            center_distance_histogram=np.zeros(0, dtype=[("window_id", "i4")]),
            warnings=(),
        )

    monkeypatch.setattr(generic_mod, "build_chaser_epoch_behavior_summary_result", fake_build)
    monkeypatch.setattr(
        generic_mod,
        "write_chaser_epoch_behavior_summary_component",
        lambda *args, **kwargs: writes.append(args),
    )

    rows = mod.run_for_targets(
        [{"recording_id": "recording_GoodCopBadCop", "zarr_path": str(zarr_path)}],
        chaser_distance_run="latest",
        component_name="kinematics_bouts_v1",
        swim_bout_run="latest",
        track_kinematics_run=None,
        track_kinematics_scope="offline",
        track_id=None,
        speed_level="filtered",
        apply=False,
        overwrite=False,
    )

    assert writes == []
    assert rows[0]["status"] == "dry_run"
    assert rows[0]["epoch_behavior_summary_path"] is None


def test_run_for_targets_rejects_missing_authoritative_speed_before_processing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        generic_mod,
        "build_chaser_epoch_behavior_summary_result",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="requires an explicit speed_level"):
        mod.run_for_targets(
            [{"recording_id": "recording", "zarr_path": str(tmp_path / "a.zarr")}],
            chaser_distance_run="latest",
            component_name="kinematics_bouts_v2",
            swim_bout_run="latest",
            track_kinematics_run=None,
            track_kinematics_scope="offline",
            track_id=0,
            speed_level=None,
            execution_mode=AUTHORITATIVE_EXECUTION_MODE,
        )

    assert calls == []


def test_run_for_targets_legacy_mode_uses_only_explicit_legacy_writer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "legacy.zarr"
    zarr_path.mkdir()
    writes: list[str] = []

    def fake_build(path: Path, **kwargs):
        assert kwargs["execution_mode"] == LEGACY_EXECUTION_MODE
        return SimpleNamespace(
            chaser_distance_run_name="chaser_distance_1",
            source_swim_bout_run=None,
            source_track_kinematics_run=None,
            per_epoch_fish=np.zeros(
                1,
                dtype=[
                    ("bout_count", "i8"),
                    ("inter_bout_interval_count", "i8"),
                ],
            ),
            per_epoch_chaser=np.zeros(0, dtype=[("window_id", "i4")]),
            per_epoch_bouts=np.zeros(0, dtype=[("window_id", "i4")]),
            per_epoch_bout_histograms=np.zeros(0, dtype=[("window_id", "i4")]),
            per_epoch_inter_bout_interval_histograms=np.zeros(
                0, dtype=[("window_id", "i4")]
            ),
            center_distance_histogram=np.zeros(0, dtype=[("window_id", "i4")]),
            warnings=("track_kinematics_unavailable: compatibility fixture",),
        )

    monkeypatch.setattr(
        generic_mod,
        "build_chaser_epoch_behavior_summary_result",
        fake_build,
    )
    monkeypatch.setattr(
        generic_mod,
        "write_chaser_epoch_behavior_summary_component",
        lambda *_args, **_kwargs: writes.append("authoritative"),
    )
    monkeypatch.setattr(
        generic_mod,
        "write_legacy_chaser_epoch_behavior_summary_component",
        lambda *_args, **_kwargs: writes.append("legacy") or "legacy/path",
    )

    rows = mod.run_for_targets(
        [{"recording_id": "recording", "zarr_path": str(zarr_path)}],
        chaser_distance_run="latest",
        component_name="kinematics_bouts_v1",
        swim_bout_run="latest",
        track_kinematics_run=None,
        track_kinematics_scope="offline",
        track_id=0,
        speed_level=None,
        execution_mode=LEGACY_EXECUTION_MODE,
        apply=True,
    )

    assert writes == ["legacy"]
    assert rows[0]["status"] == "complete"
    assert rows[0]["execution_mode"] == LEGACY_EXECUTION_MODE


def test_cli_requires_speed_level_for_authoritative_mode() -> None:
    with pytest.raises(SystemExit) as exc_info:
        generic_mod.main(["--zarr", "/tmp/not-opened.zarr"])
    assert exc_info.value.code == 2
