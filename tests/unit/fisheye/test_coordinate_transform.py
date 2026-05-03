"""Tests for fisheye.shared.coordinate_transform."""

import numpy as np
import pytest
import zarr

from fisheye.shared.coordinate_transform import (
    load_calibration_transform,
    projector_px_to_mm,
    projector_to_camera_mm,
    projector_to_camera_px,
    resolve_concentric_center_mm,
    visual_angle_deg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calibration_zarr(
    *,
    pixel_to_mm=5.8,
    pixels_per_mm_projector=0.44,
    z_eff_mm=10.4,
    arena_center_x_px=2256.0,
    arena_center_y_px=2256.0,
    homography=None,
):
    """Create a minimal zarr with analysis/calibration/ populated."""
    root = zarr.open_group("memory://", mode="w")
    calib = root.require_group("analysis/calibration")
    if pixel_to_mm is not None:
        calib.attrs["pixel_to_mm"] = pixel_to_mm
    if pixels_per_mm_projector is not None:
        calib.attrs["pixels_per_mm_projector"] = pixels_per_mm_projector
    if z_eff_mm is not None:
        calib.attrs["z_eff_mm"] = z_eff_mm
    if arena_center_x_px is not None:
        calib.attrs["arena_center_x_px"] = arena_center_x_px
    if arena_center_y_px is not None:
        calib.attrs["arena_center_y_px"] = arena_center_y_px
    if homography is not None:
        calib.create_array(
            "homography_matrix",
            shape=(3, 3),
            dtype="float64",
            fill_value=0.0,
            overwrite=True,
        )
        calib["homography_matrix"][:] = homography
    return root


# ---------------------------------------------------------------------------
# load_calibration_transform
# ---------------------------------------------------------------------------

class TestLoadCalibrationTransform:
    def test_loads_all_fields(self):
        root = _make_calibration_zarr()
        cal = load_calibration_transform(root)
        assert cal["pixel_to_mm"] == pytest.approx(5.8)
        assert cal["pixels_per_mm_projector"] == pytest.approx(0.44)
        assert cal["z_eff_mm"] == pytest.approx(10.4)
        assert cal["arena_center_px"] == pytest.approx((2256.0, 2256.0))
        assert cal["homography"] is None

    def test_missing_projector_calibration(self):
        root = _make_calibration_zarr(pixels_per_mm_projector=None)
        cal = load_calibration_transform(root)
        assert cal["pixels_per_mm_projector"] is None
        # Other fields still present.
        assert cal["pixel_to_mm"] == pytest.approx(5.8)

    def test_missing_z_eff(self):
        root = _make_calibration_zarr(z_eff_mm=None)
        cal = load_calibration_transform(root)
        assert cal["z_eff_mm"] is None

    def test_homography_loaded(self):
        H = np.eye(3) * 12.6
        root = _make_calibration_zarr(homography=H)
        cal = load_calibration_transform(root)
        np.testing.assert_allclose(cal["homography"], H)

    def test_root_calibration_pixels_per_mm_fallback_inverts_to_pixel_to_mm(self):
        root = zarr.open_group("memory://", mode="w")
        calib = root.require_group("calibration")
        calib.attrs["pixels_per_mm"] = 50.0

        cal = load_calibration_transform(root)

        assert cal["pixel_to_mm"] == pytest.approx(0.02)

    def test_stimulus_run_calibration_fallback(self):
        root = zarr.open_group("memory://", mode="w")
        stim_parent = root.require_group("analysis/stimulus_runs")
        stim_parent.attrs["latest"] = "stimulus_test"
        cam = stim_parent.require_group("stimulus_test/calibration/2010093")
        cam.attrs["pixels_per_mm_camera"] = 25.0
        cam.attrs["pixels_per_mm_projector"] = 4.0
        H = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]])
        cam.create_array("homography_matrix", data=H)

        cal = load_calibration_transform(root)

        assert cal["pixel_to_mm"] == pytest.approx(0.04)
        assert cal["pixels_per_mm_projector"] == pytest.approx(4.0)
        np.testing.assert_allclose(cal["homography"], H)

    def test_empty_zarr_returns_nones(self):
        root = zarr.open_group("memory://", mode="w")
        cal = load_calibration_transform(root)
        assert all(v is None for v in cal.values())


# ---------------------------------------------------------------------------
# projector_px_to_mm
# ---------------------------------------------------------------------------

class TestProjectorPxToMm:
    def test_scalar_conversion(self):
        # 10 px at 0.5 px/mm = 20 mm
        result = projector_px_to_mm(np.array(10.0), 0.5)
        assert result == pytest.approx(20.0)

    def test_array_conversion(self):
        px = np.array([0.0, 0.44, 4.4])
        result = projector_px_to_mm(px, 0.44)
        np.testing.assert_allclose(result, [0.0, 1.0, 10.0])

    def test_preserves_shape(self):
        px = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = projector_px_to_mm(px, 1.0)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# visual_angle_deg
# ---------------------------------------------------------------------------

class TestVisualAngleDeg:
    def test_zero_radius(self):
        angle = visual_angle_deg(np.array(0.0), z_eff_mm=10.0)
        assert angle == pytest.approx(0.0)

    def test_known_angle(self):
        # radius_mm = z_eff_mm → θ = 2*arctan(1) = 90°
        angle = visual_angle_deg(np.array(10.0), z_eff_mm=10.0)
        assert angle == pytest.approx(90.0)

    def test_small_angle_approximation(self):
        # For small radius, θ ≈ 2*radius/z_eff (in radians)
        # radius=0.1mm, z=10mm → θ ≈ 2*0.1/10 = 0.02 rad ≈ 1.146°
        angle = visual_angle_deg(np.array(0.1), z_eff_mm=10.0)
        approx_deg = np.degrees(2.0 * 0.1 / 10.0)
        assert abs(angle - approx_deg) < 0.01

    def test_array_input(self):
        radii = np.array([0.0, 5.0, 10.0])
        angles = visual_angle_deg(radii, z_eff_mm=10.0)
        assert angles.shape == (3,)
        assert angles[0] == pytest.approx(0.0)
        assert angles[2] == pytest.approx(90.0)
        # Monotonically increasing.
        assert np.all(np.diff(angles) > 0)

    def test_typical_loom_range(self):
        # start_radius = 2px, end_radius = 50px, pixels_per_mm_projector = 0.44
        # At z_eff = 10.4mm
        start_mm = 2.0 / 0.44
        end_mm = 50.0 / 0.44
        start_angle = visual_angle_deg(np.array(start_mm), z_eff_mm=10.4)
        end_angle = visual_angle_deg(np.array(end_mm), z_eff_mm=10.4)
        # Start should be small, end should be large.
        assert 0 < start_angle < 90
        assert end_angle > start_angle


# ---------------------------------------------------------------------------
# projector_to_camera (existing functions, basic smoke tests)
# ---------------------------------------------------------------------------

class TestProjectorToCamera:
    def test_identity_homography(self):
        H = np.eye(3)
        pts = np.array([[100.0, 200.0]])
        result = projector_to_camera_px(pts, H)
        np.testing.assert_allclose(result, [[100.0, 200.0]])

    def test_scale_homography(self):
        H = np.diag([12.6, 12.6, 1.0])
        pts = np.array([179.0, 179.0])  # center of 358x358
        result = projector_to_camera_px(pts, H)
        np.testing.assert_allclose(result, [179.0 * 12.6, 179.0 * 12.6])

    def test_to_mm(self):
        H = np.eye(3)
        pts = np.array([100.0, 200.0])
        pixel_to_mm = 0.5
        result = projector_to_camera_mm(pts, H, pixel_to_mm)
        np.testing.assert_allclose(result, [50.0, 100.0])
