from __future__ import annotations

import pytest
import zarr

from fisheye.shared.dish_mask_boundary import (
    DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT,
    apply_dish_mask_boundary_tolerance,
    resolve_dish_mask_boundary_tolerance,
)


def _root() -> zarr.Group:
    root = zarr.group(store=zarr.storage.MemoryStore())
    root.require_group("raw_video").attrs.update(
        {"source_video_height": 4512, "source_video_width": 4512}
    )
    return root


def test_resolve_half_mm_from_canonical_camera_calibration() -> None:
    root = _root()
    root.require_group("analysis/calibration").attrs["pixels_per_mm_camera"] = 50.0

    result = resolve_dish_mask_boundary_tolerance(
        root,
        source_group=None,
    )

    assert result["contract"] == DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT
    assert result["requested_mm"] == pytest.approx(0.5)
    assert result["tolerance_source_px"] == pytest.approx(25.0)
    assert result["tolerance_norm_x"] == pytest.approx(25.0 / 4512.0)
    assert result["calibration_source"] == (
        "analysis/calibration.attrs.pixels_per_mm_camera"
    )


def test_apply_tolerance_preserves_fitted_circle_geometry() -> None:
    effective = apply_dish_mask_boundary_tolerance(
        {
            "shape": "circle",
            "center_norm": [0.5, 0.5],
            "radius_norm_x": 0.4,
            "radius_norm_y": 0.4,
            "mask_image_shape_hw": [640.0, 640.0],
        },
        {
            "contract": DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT,
            "requested_mm": 0.5,
            "tolerance_norm_x": 25.0 / 4512.0,
            "tolerance_norm_y": 25.0 / 4512.0,
        },
    )

    assert effective["base_radius_norm_x"] == pytest.approx(0.4)
    assert effective["radius_norm_x"] == pytest.approx(0.4 + 25.0 / 4512.0)
    assert effective["boundary_tolerance"]["effective_tolerance_mask_px_xy"] == pytest.approx(
        [25.0 * 640.0 / 4512.0, 25.0 * 640.0 / 4512.0]
    )


def test_positive_tolerance_fails_closed_without_camera_calibration() -> None:
    root = _root()

    with pytest.raises(ValueError, match="pixels_per_mm_camera"):
        resolve_dish_mask_boundary_tolerance(root, source_group=None)


def test_zero_tolerance_does_not_require_calibration() -> None:
    root = _root()

    result = resolve_dish_mask_boundary_tolerance(
        root,
        source_group=None,
        tolerance_mm=0.0,
    )

    assert result["enabled"] is False
    assert result["tolerance_norm_x"] == 0.0
