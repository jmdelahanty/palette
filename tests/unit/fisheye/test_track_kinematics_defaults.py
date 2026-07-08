from __future__ import annotations

import inspect

from fisheye.analysis import track_kinematics as mod


def test_track_kinematics_reviewed_defaults_are_canonical() -> None:
    signature = inspect.signature(mod.build_track_datasets)

    assert mod.DEFAULT_SMOOTH_SECONDS == 0.05
    assert mod.DEFAULT_HYSTERESIS_HIGH_PX == 4.0
    assert mod.DEFAULT_HYSTERESIS_LOW_PX == 2.0
    assert mod.DEFAULT_HYSTERESIS_MIN_FRAMES == 3
    assert mod.DEFAULT_HYSTERESIS_BAND_POLICY == "latch"
    assert mod.DEFAULT_SMOOTHING_ALIGNMENT == "causal"
    assert signature.parameters["hysteresis_band_policy"].default == mod.DEFAULT_HYSTERESIS_BAND_POLICY
    assert signature.parameters["smoothing_alignment"].default == mod.DEFAULT_SMOOTHING_ALIGNMENT
