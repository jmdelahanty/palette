"""Preservation and synthetic adversarial controls for the opt-in rim recipe."""

from __future__ import annotations

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from fisheye.diagnostics import probe_recording_dish_rim_fit as probe
from fisheye.analysis_workflows.materializers import arena_geometry_fit_review as review


def _image(*, center=(256, 256), radii=(185, 215)):
    image = np.full((512, 512), 30, dtype=np.uint8)
    # The inner contour is stronger: preference must not just maximize contrast.
    for radius, value in zip(radii, (230, 120)):
        cv2.circle(image, center, radius, value, 3, cv2.LINE_AA)
    return image


def _candidate(radius=215.0, *, x=256.0, y=256.0, name="edge"):
    return probe.CircleCandidate(name, x, y, radius, 1.0, 0.0, 100.0, 100.0)


def _fit(candidate):
    return probe.CircleFit(
        candidate.center_x_px,
        candidate.center_y_px,
        candidate.radius_px,
        candidate.angular_support_fraction,
        candidate.median_radial_gradient,
        1,
        selected_candidate_id=candidate.candidate_id,
        frozen_candidates=(candidate,),
    )


def _bundle():
    composites = {name: _image() for name in probe.WINDOW_NAMES}
    fits = {name: _fit(_candidate()) for name in probe.WINDOW_NAMES}
    source = {
        "camera_serial": "2010093",
        "image_shape_px": {"height": 512, "width": 512},
        "pixel_contract": "orange.camera.mono8.full_frame.v1",
        "video_sha256": "a" * 64,
    }
    metrics = probe.build_rim_metrics(composites, fits=fits, source=source)
    windows = {
        name: {
            "fit": fit.to_json(),
            "composite_pixel_sha256": hashlib.sha256(
                composites[name].tobytes(order="C")
            ).hexdigest(),
        }
        for name, fit in fits.items()
    }
    return metrics, source, windows


def test_legacy_circle_serialization_digest_is_unchanged():
    payload = probe.CircleFit(12.25, 13.5, 90.0, 0.8, 54.0, 2).to_json()
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "1c26b341343d17245dd7257865a1ded0c71f5be0820e349f17521bad92513252"
    )
    args = probe.build_parser().parse_args(["--video", "x", "--output-dir", "out"])
    assert args.fit_recipe == "legacy_v1"


def test_metrics_measure_support_residual_and_all_quadrants_without_refitting():
    candidate = _candidate()
    before = candidate.to_json()
    metric = probe.measure_rim_candidate_metrics(_image(), candidate)
    assert metric["geometry"] == before["geometry"]
    assert metric["angular_support_fraction"] > 0.95
    assert metric["visible_angular_fraction"] == 1.0
    assert min(metric["quadrant_support_fractions"]) > 0.95
    assert metric["radial_residual_p95_px"] < 4.1
    assert metric["longest_unsupported_arc_degrees"] < 10
    assert metric["quality_flags"] == []
    assert candidate.to_json() == before


def test_blank_image_has_no_support_not_zero_residual_success():
    metric = probe.measure_rim_candidate_metrics(
        np.zeros((512, 512), np.uint8), _candidate()
    )
    assert metric["angular_support_fraction"] == 0.0
    assert metric["longest_unsupported_arc_degrees"] == 360.0
    assert metric["radial_residual_p95_px"] is None
    assert "no_support" in metric["quality_flags"]


def test_occlusion_is_localized_and_clipping_remains_visible():
    occluded = _image()
    occluded[:256, :256] = 30
    metric = probe.measure_rim_candidate_metrics(occluded, _candidate())
    assert metric["longest_unsupported_arc_degrees"] > 70
    assert min(metric["quadrant_support_fractions"]) < 0.1
    assert "occluded_or_low_support" in metric["quality_flags"]
    clipped = probe.measure_rim_candidate_metrics(
        _image(center=(128, 256)), _candidate(x=128)
    )
    assert clipped["visible_angular_fraction"] < 0.8
    assert "clipped" in clipped["quality_flags"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True])
def test_metrics_reject_invalid_geometry(value):
    with pytest.raises(ValueError, match="finite|positive|numeric"):
        probe.measure_rim_candidate_metrics(_image(), _candidate(radius=value))


def test_metrics_reject_silent_dtype_and_coordinate_coercion():
    with pytest.raises(ValueError, match="uint8"):
        probe.measure_rim_candidate_metrics(_image().astype(float), _candidate())
    with pytest.raises(ValueError, match="native"):
        probe.measure_rim_candidate_metrics(_image(), _candidate(x=999))


def test_opt_in_prefers_supported_outer_family_but_does_not_identify_top_rim():
    fit, edge = probe.fit_dish_circle_top_rim_preferred(
        _image(), coarse_max_dimension_px=512
    )
    assert 211 < fit.radius_px < 220
    assert min(candidate.radius_px for candidate in fit.frozen_candidates) < 190
    assert (
        fit.selection_reason
        == "outermost_supported_concentric_edge_top_rim_preference_v1"
    )
    assert edge.shape == (512, 512)
    assert fit.to_json()["intended_target_feature"] == "visible_dish_top_rim_edge"
    assert (
        fit.to_json()["observed_feature_classification"]
        == "unclassified_concentric_rim_edge"
    )


def test_top_rim_recipe_refuses_no_support():
    with pytest.raises(RuntimeError, match="candidates|support"):
        probe.fit_dish_circle_top_rim_preferred(
            np.zeros((512, 512), np.uint8), coarse_max_dimension_px=512
        )


def test_temporal_metrics_preserve_all_circles_and_bind_exact_source():
    metrics, source, windows = _bundle()
    review.validate_rim_metrics(metrics, source=source, windows=windows)
    assert metrics["semantic_correspondence"] == "projected_edges_unresolved"
    assert metrics["physical_feature_identified"] is False
    assert metrics["temporal"]["center_max_pairwise_distance_px"] == 0.0
    assert metrics["temporal"]["radius_range_px"] == 0.0
    assert metrics["temporal"]["minimum_candidate_count"] == 1
    assert set(metrics["windows"]["early"]["candidates"]) == {"edge"}


def test_source_hash_rejects_nonfinite_identity_instead_of_normalizing_to_null():
    with pytest.raises(ValueError):
        review.rim_metrics_source_sha256({"fps": float("nan")})


def test_family_dropout_is_measured_without_equating_physical_edges():
    metrics, source, windows = _bundle()
    second = _candidate(radius=185, name="inner_gradient")
    windows["early"]["fit"]["frozen_candidates"].append(second.to_json())
    windows["early"]["fit"]["candidate_count"] = 2
    metrics["windows"]["early"]["candidates"][second.candidate_id] = (
        probe.measure_rim_candidate_metrics(_image(), second)
    )
    metrics["temporal"] = review.summarize_rim_metric_temporal(metrics["windows"])
    assert metrics["temporal"]["rim_family_radius_hausdorff_distance_px"] == 30.0
    review.validate_rim_metrics(metrics, source=source, windows=windows)


def test_temporal_drift_is_measured_not_hidden_in_consensus():
    composites = {
        name: _image(center=(256 + index * 5, 256))
        for index, name in enumerate(probe.WINDOW_NAMES)
    }
    fits = {
        name: _fit(_candidate(x=256 + index * 5, radius=215 + index * 2))
        for index, name in enumerate(probe.WINDOW_NAMES)
    }
    metrics = probe.build_rim_metrics(
        composites,
        fits=fits,
        source={"camera_serial": "c", "image_shape_px": {"height": 512, "width": 512}},
    )
    assert metrics["temporal"]["center_max_pairwise_distance_px"] == 10.0
    assert metrics["temporal"]["radius_range_px"] == 4.0
    assert "temporal_variation_observed" in metrics["quality_flags"]


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "composite",
        "geometry",
        "nan",
        "summary",
        "missing_window",
        "selected_id",
        "fake_semantics",
    ],
)
def test_validator_rejects_stale_tampered_or_incomplete_metrics(mutation):
    metrics, source, windows = _bundle()
    metrics = copy.deepcopy(metrics)
    if mutation == "source":
        source["camera_serial"] = "wrong_camera"
    elif mutation == "composite":
        metrics["windows"]["early"]["composite_pixel_sha256"] = "b" * 64
    elif mutation == "geometry":
        metrics["windows"]["early"]["candidates"]["edge"]["geometry"]["radius_px"] += 1
    elif mutation == "nan":
        metrics["windows"]["early"]["candidates"]["edge"][
            "angular_support_fraction"
        ] = float("nan")
    elif mutation == "summary":
        metrics["temporal"]["radius_range_px"] = 9.0
    elif mutation == "missing_window":
        del metrics["windows"]["late"]
    elif mutation == "selected_id":
        metrics["windows"]["early"]["selected_candidate_id"] = "missing"
    else:
        metrics["physical_feature_identified"] = True
    with pytest.raises(ValueError):
        review.validate_rim_metrics(metrics, source=source, windows=windows)
