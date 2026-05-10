from __future__ import annotations

import inspect
import sys

from fisheye.analysis import detect_bouts_multi_level as mod


def test_swim_bout_layout_default_constant_controls_function_default() -> None:
    signature = inspect.signature(mod.detect_and_save_bouts)

    assert mod.SWIM_BOUT_LAYOUT_DEFAULT == mod.SWIM_BOUT_LAYOUT_HIERARCHICAL_V1
    assert signature.parameters["layout"].default == mod.SWIM_BOUT_LAYOUT_DEFAULT


def test_swim_bout_cli_uses_layout_default_constant(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_detect_and_save_bouts(**kwargs):
        captured.update(kwargs)
        return "fake_bouts"

    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    monkeypatch.setattr(mod, "detect_and_save_bouts", fake_detect_and_save_bouts)
    monkeypatch.setattr(sys, "argv", ["detect_bouts_multi_level", str(zarr_path)])

    assert mod.main() == 0
    assert captured["layout"] == mod.SWIM_BOUT_LAYOUT_DEFAULT
