from __future__ import annotations

from fisheye.visualization.visualize_crops import _cycle_intended_use


def test_cycle_intended_use_round_trip() -> None:
    assert _cycle_intended_use("training") == "full_recording"
    assert _cycle_intended_use("full_recording") == "training"


def test_cycle_intended_use_unknown_falls_back_to_training() -> None:
    assert _cycle_intended_use("unknown") == "training"
