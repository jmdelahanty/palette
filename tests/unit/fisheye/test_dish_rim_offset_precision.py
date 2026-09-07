"""Real-canary regression: known radial sample offsets must stay exact."""

import hashlib

import numpy as np

from fisheye.diagnostics import probe_recording_dish_rim_fit as probe


def test_legacy_two_array_radial_evidence_bytes_are_unchanged():
    evidence = probe._radial_evidence(
        np.arange(4096, dtype=np.float32).reshape(64, 64),
        (31.25, 32.75, 20.5),
        radial_band_px=4.0,
        angle_count=720,
    )
    assert len(evidence) == 2
    assert hashlib.sha256(
        evidence[0].tobytes() + evidence[1].tobytes()
    ).hexdigest() == (
        "f8741e0af57235058987c30dfa5f0ea44988e25f2b9a567c2a50b18d44d782db"
    )


def test_radial_evidence_can_return_exact_selected_grid_offsets():
    points, peaks, offsets = probe._radial_evidence(
        np.ones((512, 512), dtype=np.float32),
        (256.0, 256.0, 215.0),
        radial_band_px=4.0,
        angle_count=720,
        return_radial_offsets=True,
    )
    assert np.all(peaks == 1.0)
    assert np.all(offsets == -4.0)
    # atan/trig float32 roundoff affects reconstructed point distances, not
    # the exact offsets chosen from the integer-spaced radial sampling grid.
    reconstructed = np.abs(np.hypot(points[:, 0] - 256, points[:, 1] - 256) - 215)
    assert np.percentile(reconstructed, 95) > 4.0


def test_fixed_band_measurement_does_not_turn_roundoff_into_ineligibility():
    candidate = probe.CircleCandidate("exact_band", 256.0, 256.0, 215.0, 1, 0, 1, 1)
    metric = probe._measure_rim_candidate_gradient(
        np.ones((512, 512), dtype=np.float32), candidate
    )
    assert metric["angular_support_fraction"] == 1.0
    assert metric["radial_residual_p95_px"] == 4.0
    assert metric["median_absolute_radial_offset_px"] == 4.0
    assert (
        metric["radial_residual_p95_px"]
        <= probe.TOP_RIM_PARAMETERS["preferred_max_radial_residual_p95_px"]
    )
    assert (
        probe.TOP_RIM_PARAMETERS["radial_offset_measurement_method"]
        == "signed_sampling_grid_offset_v1"
    )
