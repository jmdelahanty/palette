"""Real-canary regression: known radial sample offsets must stay exact."""

import cv2
import numpy as np
import pytest

from fisheye.diagnostics import probe_recording_dish_rim_fit as probe


# Frozen from probe_recording_dish_rim_fit._radial_evidence at exact commit
# ac73a61067118fc1a41699267dfa376a15f4bac6. Keep this historical algorithm
# independent of the current implementation: floating-point bytes can vary
# across NumPy/OpenCV builds and CPU dispatch, so compare within one runtime.
def _legacy_radial_evidence_ac73a610(
    gradient: np.ndarray,
    circle: tuple[float, float, float],
    *,
    radial_band_px: float,
    angle_count: int = 1440,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, radius = circle
    angles = np.linspace(
        0.0, 2.0 * np.pi, angle_count, endpoint=False, dtype=np.float32
    )
    offsets = np.linspace(-radial_band_px, radial_band_px, 2 * int(radial_band_px) + 1)
    radii = radius + offsets[:, None]
    map_x = (cx + radii * np.cos(angles)[None, :]).astype(np.float32)
    map_y = (cy + radii * np.sin(angles)[None, :]).astype(np.float32)
    sampled = cv2.remap(
        gradient,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    best_rows = np.argmax(sampled, axis=0)
    columns = np.arange(angle_count)
    peaks = sampled[best_rows, columns]
    peak_radii = radius + offsets[best_rows]
    points = np.column_stack(
        (cx + peak_radii * np.cos(angles), cy + peak_radii * np.sin(angles))
    )
    return points.astype(np.float64), peaks.astype(np.float64)


@pytest.mark.parametrize(
    "gradient,circle,band,angle_count",
    [
        pytest.param(
            np.arange(4096, dtype=np.float32).reshape(64, 64),
            (31.25, 32.75, 20.5),
            4.0,
            720,
            id="original_ci_input",
        ),
        pytest.param(
            np.ones((64, 64), dtype=np.float32),
            (32.0, 32.0, 20.0),
            4.0,
            720,
            id="flat_band_ties",
        ),
        pytest.param(
            np.random.default_rng(1729).random((64, 64), dtype=np.float32),
            (29.7, 34.3, 22.8),
            2.0,
            1440,
            id="seeded_gradient",
        ),
        pytest.param(
            np.arange(1024, dtype=np.float32).reshape(32, 32),
            (1.25, 3.5, 40.0),
            4.0,
            360,
            id="out_of_bounds",
        ),
    ],
)
def test_legacy_two_array_radial_evidence_bytes_are_unchanged(
    gradient, circle, band, angle_count
):
    expected = _legacy_radial_evidence_ac73a610(
        gradient, circle, radial_band_px=band, angle_count=angle_count
    )
    evidence = probe._radial_evidence(
        gradient, circle, radial_band_px=band, angle_count=angle_count
    )
    assert len(evidence) == len(expected) == 2
    for actual, legacy in zip(evidence, expected, strict=True):
        assert actual.shape == legacy.shape
        assert actual.dtype == legacy.dtype == np.dtype(np.float64)
        assert actual.tobytes(order="C") == legacy.tobytes(order="C")


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
