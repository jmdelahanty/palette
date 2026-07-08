from __future__ import annotations

import inspect
import sys

from fisheye.analysis import detect_bouts_multi_level as mod


def test_swim_bout_layout_default_constant_controls_function_default() -> None:
    signature = inspect.signature(mod.detect_and_save_bouts)

    assert mod.SWIM_BOUT_LAYOUT_DEFAULT == mod.SWIM_BOUT_LAYOUT_COMPACT_V2
    assert signature.parameters["layout"].default == mod.SWIM_BOUT_LAYOUT_DEFAULT
    assert signature.parameters["method"].default == mod.DEFAULT_DETECTION_METHOD
    assert signature.parameters["default_level"].default == mod.DEFAULT_SWIM_BOUT_LEVEL
    assert signature.parameters["exponential_tau_s"].default == mod.DEFAULT_EXPONENTIAL_TAU_S
    assert (
        signature.parameters["min_peak_prominence_mm_s"].default
        == mod.DEFAULT_MIN_PEAK_PROMINENCE_MM_S
    )
    assert signature.parameters["min_peak_distance_s"].default == mod.DEFAULT_MIN_PEAK_DISTANCE_S
    assert signature.parameters["peak_width_rel_height"].default == mod.DEFAULT_PEAK_WIDTH_REL_HEIGHT


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
    assert captured["method"] == mod.DEFAULT_DETECTION_METHOD
    assert captured["default_level"] == mod.DEFAULT_SWIM_BOUT_LEVEL
    assert captured["exponential_tau_s"] == mod.DEFAULT_EXPONENTIAL_TAU_S
    assert captured["min_peak_prominence_mm_s"] == mod.DEFAULT_MIN_PEAK_PROMINENCE_MM_S
    assert captured["min_peak_distance_s"] == mod.DEFAULT_MIN_PEAK_DISTANCE_S
    assert captured["peak_width_rel_height"] == mod.DEFAULT_PEAK_WIDTH_REL_HEIGHT
