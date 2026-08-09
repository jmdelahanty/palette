from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.arena_geometry import (
    GEOMETRY_STATUS_DISH_MASK,
    GEOMETRY_STATUS_NOMINAL_CIRCLE,
    ArenaGeometry,
    detection_frame_size_px,
    out_of_bounds_fraction,
    out_of_bounds_notes,
    read_dish_mask_circle,
    require_dish_mask_arena_geometry,
    resolve_arena_geometry,
)


PPM = 4.0                 # px per mm in the arena-canvas frame
FRAME = 4512.0            # detection frame is square
ORIGIN = (100.0, 50.0)    # arena origin in canvas px


def _make_archive(
    tmp_path: Path,
    *,
    dish_mask: dict | None,
    nominal_center_px: tuple[float, float] = (172.0, 172.0),
    nominal_radius_px: float = 160.0,
    homography: np.ndarray | None = None,
    name: str = "geom.zarr",
) -> tuple[Path, zarr.Group]:
    """Archive with BOTH circles present, so tests can prove which one wins."""

    z = tmp_path / name
    root = zarr.open_group(str(z), mode="w")

    if dish_mask is not None:
        root.require_group("analysis_metadata").attrs["dish_mask"] = dish_mask

    calibration = root.require_group("analysis/calibration")
    # Identity-scaled homography: camera px -> canvas px at `scale`, then the arena origin is
    # subtracted downstream. Keeps the expected geometry analytically checkable.
    H = np.eye(3) if homography is None else np.asarray(homography, dtype=np.float64)
    calibration.create_array("homography_matrix", data=H, overwrite=True)

    detect = root.require_group("detect_runs/detect_1")
    detect.attrs["source_video_width"] = FRAME
    detect.attrs["source_video_height"] = FRAME

    stim = root.require_group("analysis/stimulus_runs/stimulus_1/calibration/arena_geometry")
    stim.attrs.update(
        {
            "arena_origin_in_canvas_x_px": ORIGIN[0],
            "arena_origin_in_canvas_y_px": ORIGIN[1],
            "arena_region_width_px": 348.0,
            "arena_region_height_px": 348.0,
            "experimental_area_shape": "CIRCLE",
            "experimental_area_center_x_px": nominal_center_px[0],
            "experimental_area_center_y_px": nominal_center_px[1],
            "experimental_area_radius_px": nominal_radius_px,
        }
    )
    run = root.require_group("analysis/chaser_distance_runs/run_1")
    run.attrs["source_stimulus_path"] = "analysis/stimulus_runs/stimulus_1"
    run.attrs["pixels_per_mm_projector"] = PPM
    return z, run


def _mask(center_px: tuple[float, float], radius_px: float, *, image: float = 640.0) -> dict:
    """A dish mask exactly as Citrus writes it: fitted on the downsampled image."""

    return {
        "shape": "circle",
        "method": "hough_circle",
        "detected_circle": {"center": [center_px[0], center_px[1]], "radius": radius_px},
        "metrics": {"image_shape": [image, image]},
    }


# --------------------------------------------------------------------------------------
# Reading and rescaling the mask
# --------------------------------------------------------------------------------------


def test_dish_mask_is_read_as_normalized_coordinates(tmp_path: Path) -> None:
    """The mask is tuned on a 640px downsample but detections live at 4512. Normalizing is
    what makes the two comparable -- the same thing refine_detect does."""

    z, _run = _make_archive(tmp_path, dish_mask=_mask((316.0, 318.0), 304.0))
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    circle = read_dish_mask_circle(root)

    assert circle == pytest.approx(
        {"center_norm_x": 316 / 640, "center_norm_y": 318 / 640, "radius_norm": 304 / 640}
    )
    assert detection_frame_size_px(root) == (FRAME, FRAME)


def test_dish_mask_falls_back_to_metrics_center_norm(tmp_path: Path) -> None:
    z, _run = _make_archive(
        tmp_path,
        dish_mask={
            "shape": "circle",
            "metrics": {"image_shape": [640.0, 640.0], "center_norm": [0.5, 0.5], "radius_norm": 0.4},
        },
    )
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    assert read_dish_mask_circle(root) == pytest.approx(
        {"center_norm_x": 0.5, "center_norm_y": 0.5, "radius_norm": 0.4}
    )


def test_non_circular_dish_mask_is_ignored(tmp_path: Path) -> None:
    z, _run = _make_archive(tmp_path, dish_mask={"shape": "rectangle", "rectangle": {"roi": [0, 0, 10, 10]}})
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    assert read_dish_mask_circle(root) is None


# --------------------------------------------------------------------------------------
# The substitution that matters
# --------------------------------------------------------------------------------------


def test_dish_mask_wins_over_the_nominal_circle(tmp_path: Path) -> None:
    """Both circles present and they disagree. The mask must win -- this is the entire bug."""

    # Mask centred at 0.5,0.5 of the frame with radius 0.25 of it. Under the identity
    # homography the canvas point is the camera point, minus the arena origin.
    z, run = _make_archive(
        tmp_path,
        dish_mask=_mask((320.0, 320.0), 160.0),   # -> norm 0.5, 0.5, r 0.25
        nominal_center_px=(172.0, 172.0),
        nominal_radius_px=160.0,
    )
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    geometry, notes = resolve_arena_geometry(root, root[run.path], pixels_per_mm=PPM)

    assert geometry.status == GEOMETRY_STATUS_DISH_MASK
    assert geometry.source == "analysis_metadata.dish_mask"
    assert notes == []

    expected_cx = 0.5 * FRAME - ORIGIN[0]
    expected_cy = 0.5 * FRAME - ORIGIN[1]
    assert geometry.center_x_px == pytest.approx(expected_cx, abs=1e-3)
    assert geometry.center_y_px == pytest.approx(expected_cy, abs=1e-3)
    assert geometry.radius_px == pytest.approx(0.25 * FRAME, abs=1e-3)
    assert geometry.fit_residual_px == pytest.approx(0.0, abs=1e-6)

    # ...and it is emphatically NOT the nominal circle.
    assert geometry.radius_px != pytest.approx(160.0)


def test_falls_back_to_the_nominal_circle_when_no_mask_exists(tmp_path: Path) -> None:
    z, run = _make_archive(tmp_path, dish_mask=None)
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    geometry, notes = resolve_arena_geometry(root, root[run.path], pixels_per_mm=PPM)

    assert geometry.status == GEOMETRY_STATUS_NOMINAL_CIRCLE
    assert geometry.center_x_px == pytest.approx(172.0)
    assert geometry.radius_px == pytest.approx(160.0)
    # The fallback must be announced. A caller that silently gets the wrong circle is the bug.
    assert any(n.startswith("arena_geometry_fallback_to_nominal") for n in notes)


def test_virtual_control_geometry_requires_the_fitted_dish_mask(tmp_path: Path) -> None:
    z, run = _make_archive(
        tmp_path,
        dish_mask=_mask((320.0, 320.0), 160.0),
    )
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)

    geometry, notes = require_dish_mask_arena_geometry(
        root,
        root[run.path],
        pixels_per_mm=PPM,
        consumer="test_virtual_controls",
    )

    assert geometry.status == GEOMETRY_STATUS_DISH_MASK
    assert geometry.source == "analysis_metadata.dish_mask"
    assert notes == []


def test_virtual_control_geometry_rejects_nominal_fallback(tmp_path: Path) -> None:
    z, run = _make_archive(tmp_path, dish_mask=None)
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)

    with pytest.raises(
        ValueError,
        match="requires fitted analysis_metadata.dish_mask geometry",
    ):
        require_dish_mask_arena_geometry(
            root,
            root[run.path],
            pixels_per_mm=PPM,
            consumer="test_virtual_controls",
        )


@pytest.mark.parametrize("pixels_per_mm", [0.0, -1.0, math.nan])
def test_virtual_control_geometry_rejects_invalid_scale(
    tmp_path: Path,
    pixels_per_mm: float,
) -> None:
    z, run = _make_archive(
        tmp_path,
        dish_mask=_mask((320.0, 320.0), 160.0),
    )
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)

    with pytest.raises(ValueError, match="requires a positive pixels_per_mm_projector"):
        require_dish_mask_arena_geometry(
            root,
            root[run.path],
            pixels_per_mm=pixels_per_mm,
            consumer="test_virtual_controls",
        )


def test_homography_scaling_is_applied(tmp_path: Path) -> None:
    """The mask lives in camera px; the fish lives in canvas px. A mask that skipped the
    homography would be off by exactly the homography's scale."""

    scale = 0.05
    H = np.diag([scale, scale, 1.0])
    z, run = _make_archive(tmp_path, dish_mask=_mask((320.0, 320.0), 160.0), homography=H)
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    geometry, _notes = resolve_arena_geometry(root, root[run.path], pixels_per_mm=PPM)

    assert geometry.radius_px == pytest.approx(scale * 0.25 * FRAME, abs=1e-3)
    assert geometry.center_x_px == pytest.approx(scale * 0.5 * FRAME - ORIGIN[0], abs=1e-3)


def test_projective_homography_that_breaks_the_circle_is_rejected(tmp_path: Path) -> None:
    """A homography is projective, so a circle maps to a conic. If it is badly non-circular we
    must not silently pretend it is a circle -- fall back and say why."""

    H = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.002, 0.0015, 1.0]])
    z, run = _make_archive(tmp_path, dish_mask=_mask((320.0, 320.0), 160.0), homography=H)
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    geometry, notes = resolve_arena_geometry(root, root[run.path], pixels_per_mm=PPM)

    assert geometry.status == GEOMETRY_STATUS_NOMINAL_CIRCLE
    assert any(n.startswith("dish_mask_projection_not_circular") for n in notes)


def test_missing_homography_falls_back_with_a_note(tmp_path: Path) -> None:
    z, run = _make_archive(tmp_path, dish_mask=_mask((320.0, 320.0), 160.0))
    root = zarr.open_group(str(z), mode="a", use_consolidated=False)
    del root["analysis/calibration"]["homography_matrix"]

    geometry, notes = resolve_arena_geometry(root, root[run.path], pixels_per_mm=PPM)
    assert geometry.status == GEOMETRY_STATUS_NOMINAL_CIRCLE
    assert any(n.startswith("dish_mask_unusable:no_homography_matrix") for n in notes)


def test_no_geometry_at_all_is_a_clear_error(tmp_path: Path) -> None:
    z, run = _make_archive(tmp_path, dish_mask=None)
    root = zarr.open_group(str(z), mode="a", use_consolidated=False)
    stim = root["analysis/stimulus_runs/stimulus_1/calibration/arena_geometry"]
    for key in ("experimental_area_shape", "experimental_area_center_x_px",
                "experimental_area_center_y_px", "experimental_area_radius_px",
                "arena_region_width_px", "arena_region_height_px"):
        del stim.attrs[key]

    with pytest.raises(ValueError, match="Unable to resolve arena geometry"):
        resolve_arena_geometry(root, root[run.path], pixels_per_mm=PPM)


# --------------------------------------------------------------------------------------
# Out-of-bounds: the geometry is wrong, not the fish
# --------------------------------------------------------------------------------------


def _geom(cx: float, cy: float, r: float) -> ArenaGeometry:
    return ArenaGeometry("dish_mask", "test", 2 * r, 2 * r, "circle", cx, cy, r)


def test_out_of_bounds_is_zero_for_a_correct_circle() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    radius = 40.0 * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * math.pi
    pts = np.stack([100 + radius * np.cos(theta), 100 + radius * np.sin(theta)], axis=1)

    assert out_of_bounds_fraction(pts, None, _geom(100, 100, 42.0)) == pytest.approx(0.0)
    assert out_of_bounds_notes(pts, None, _geom(100, 100, 42.0)) == []


def test_out_of_bounds_is_loud_for_a_circle_that_is_too_small() -> None:
    """This is the failure that inverted thigmotaxis: a wall-hugging fish read as 'outside the
    arena' by a radius that was 2.4mm short. It must warn, not stay silent."""

    theta = np.linspace(0, 2 * math.pi, 1000, endpoint=False)
    pts = np.stack([100 + 41.0 * np.cos(theta), 100 + 41.0 * np.sin(theta)], axis=1)  # on the real wall

    too_small = _geom(100, 100, 38.0)

    assert out_of_bounds_fraction(pts, None, too_small) == pytest.approx(1.0)
    notes = out_of_bounds_notes(pts, None, too_small, label="fish")
    assert notes and notes[0].startswith("arena_geometry_out_of_bounds:fish:")

    correct = _geom(100, 100, 42.0)
    assert out_of_bounds_fraction(pts, None, correct) == pytest.approx(0.0)
    assert out_of_bounds_notes(pts, None, correct) == []


def test_out_of_bounds_respects_the_validity_mask() -> None:
    pts = np.asarray([[100.0, 100.0], [999.0, 999.0]])
    valid = np.asarray([True, False])
    assert out_of_bounds_fraction(pts, valid, _geom(100, 100, 10.0)) == pytest.approx(0.0)
    assert out_of_bounds_fraction(pts, None, _geom(100, 100, 10.0)) == pytest.approx(0.5)
