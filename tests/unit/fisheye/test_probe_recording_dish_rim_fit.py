from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fisheye.diagnostics.probe_recording_dish_rim_fit import (
    CircleFit,
    build_window_specs,
    consensus_circle,
    fit_dish_circle,
    render_acquisition_reveal,
    temporal_median,
)


def _synthetic_rim(
    *,
    shape: tuple[int, int] = (512, 512),
    circle: tuple[int, int, int] = (251, 258, 220),
) -> np.ndarray:
    image = np.full(shape, 42, dtype=np.uint8)
    cv2.circle(image, circle[:2], circle[2], 205, 5, cv2.LINE_AA)
    cv2.circle(image, circle[:2], circle[2] - 13, 115, 3, cv2.LINE_AA)
    gradient = np.linspace(0, 18, shape[1], dtype=np.uint8)
    return cv2.add(image, np.broadcast_to(gradient, shape))


def test_build_window_specs_are_exact_ordered_and_disjoint() -> None:
    specs = build_window_specs(frame_count=140_000, fps=100.0)

    assert [spec.name for spec in specs] == ["early", "middle", "late"]
    assert all(len(spec.frame_indices) == 61 for spec in specs)
    assert all(
        tuple(sorted(spec.frame_indices)) == spec.frame_indices for spec in specs
    )
    assert not set(specs[0].frame_indices) & set(specs[1].frame_indices)
    assert not set(specs[1].frame_indices) & set(specs[2].frame_indices)
    assert specs[0].frame_indices[-1] - specs[0].frame_indices[0] == 500


def test_temporal_median_rejects_non_uint8_and_suppresses_transient() -> None:
    frames = np.zeros((5, 32, 32), dtype=np.uint8)
    frames[0, 4:20, 4:20] = 255
    assert not np.any(temporal_median(frames))

    with pytest.raises(ValueError, match="uint8"):
        temporal_median(np.zeros((5, 32, 32), dtype=np.float32))


def test_fit_dish_circle_uses_radial_evidence() -> None:
    expected = (251.0, 258.0, 220.0)
    fit, edge = fit_dish_circle(_synthetic_rim(), coarse_max_dimension_px=512)

    assert fit.center_x_px == pytest.approx(expected[0], abs=4.0)
    assert fit.center_y_px == pytest.approx(expected[1], abs=4.0)
    assert fit.radius_px == pytest.approx(expected[2], abs=6.0)
    assert fit.angular_support_fraction > 0.70
    assert fit.candidate_count >= 1
    assert edge.shape == (512, 512)
    assert edge.dtype == np.uint8


def test_consensus_uses_medians() -> None:
    fits = [
        CircleFit(10, 20, 30, 0.8, 100, 2),
        CircleFit(11, 19, 31, 0.9, 110, 3),
        CircleFit(400, 500, 600, 0.1, 1, 4),
    ]
    result = consensus_circle(fits)
    assert (result.center_x_px, result.center_y_px, result.radius_px) == (11, 20, 31)
    assert result.candidate_count == 9


def test_acquisition_reveal_does_not_modify_frozen_fit_report(tmp_path: Path) -> None:
    fit_report = {
        "windows": {
            name: {
                "fit": CircleFit(251, 258, 220, 0.9, 100, 1).to_json(),
            }
            for name in ("early", "middle", "late")
        }
    }
    fit_report_path = tmp_path / "fit_report.json"
    fit_report_path.write_text(json.dumps(fit_report, sort_keys=True))
    before = fit_report_path.read_bytes()
    observation = {
        "artifact_id": "dishrim_test",
        "camera": {"height": 512, "width": 512},
        "accepted_inner_rim_boundary": {
            "geometry": {
                "type": "circle",
                "center_px": {"x": 250.0, "y": 257.0},
                "radius_px": 219.0,
            }
        },
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation))
    composites = {name: _synthetic_rim() for name in ("early", "middle", "late")}

    reveal_path = render_acquisition_reveal(
        output_dir=tmp_path,
        observation_path=observation_path,
        fit_report_path=fit_report_path,
        composites=composites,
    )

    assert fit_report_path.read_bytes() == before
    reveal = json.loads(reveal_path.read_text())
    assert reveal["purpose"] == "visual_reveal_only_after_blind_palette_fit_was_frozen"
    assert reveal["files"]["early"]["delta_radius_px"] == pytest.approx(1.0)
    assert (tmp_path / "early_acquisition_reveal.png").is_file()
