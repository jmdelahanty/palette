from __future__ import annotations

import inspect

import pytest

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


@pytest.mark.parametrize(
    "coordinate_space",
    ["camera", "source_camera_image_px"],
)
def test_mm_per_pixel_resolver_uses_camera_scale_for_camera_spaces(
    coordinate_space: str,
) -> None:
    resolved = mod.resolve_mm_per_pixel_for_coordinate_space(
        coordinate_space,
        camera_mm_per_pixel=0.02,
        pixels_per_mm_projector=5.0,
    )

    assert resolved == pytest.approx(0.02)


@pytest.mark.parametrize(
    "coordinate_space",
    [
        "texture",
        "stimulus_texture_px",
        "stimulus_canvas_px",
        "projector_px",
        "arena_relative_canvas_px",
    ],
)
def test_mm_per_pixel_resolver_inverts_projector_pixels_per_mm(
    coordinate_space: str,
) -> None:
    resolved = mod.resolve_mm_per_pixel_for_coordinate_space(
        coordinate_space,
        camera_mm_per_pixel=0.02,
        pixels_per_mm_projector=5.0,
    )

    assert resolved == pytest.approx(0.2)


def test_mm_per_pixel_resolver_never_cross_falls_back() -> None:
    with pytest.raises(ValueError, match="camera_mm_per_pixel"):
        mod.resolve_mm_per_pixel_for_coordinate_space(
            "camera",
            camera_mm_per_pixel=None,
            pixels_per_mm_projector=5.0,
        )

    with pytest.raises(ValueError, match="pixels_per_mm_projector"):
        mod.resolve_mm_per_pixel_for_coordinate_space(
            "texture",
            camera_mm_per_pixel=0.02,
            pixels_per_mm_projector=None,
        )


@pytest.mark.parametrize(
    "coordinate_space",
    [None, "", "unknown", "camera_px", 123],
)
def test_mm_per_pixel_resolver_rejects_missing_or_unsupported_space(
    coordinate_space: object,
) -> None:
    with pytest.raises(ValueError, match="coordinate_space"):
        mod.resolve_mm_per_pixel_for_coordinate_space(
            coordinate_space,
            camera_mm_per_pixel=0.02,
            pixels_per_mm_projector=5.0,
        )


@pytest.mark.parametrize(
    "pixels_per_mm_projector",
    [None, 0.0, -1.0, float("nan"), float("inf"), "invalid"],
)
def test_mm_per_pixel_resolver_rejects_invalid_projector_scale(
    pixels_per_mm_projector: object,
) -> None:
    with pytest.raises(ValueError, match="positive finite pixels_per_mm_projector"):
        mod.resolve_mm_per_pixel_for_coordinate_space(
            "stimulus_texture_px",
            camera_mm_per_pixel=0.02,
            pixels_per_mm_projector=pixels_per_mm_projector,
        )


@pytest.mark.parametrize(
    "camera_mm_per_pixel",
    [None, 0.0, -1.0, float("nan"), float("inf"), "invalid"],
)
def test_mm_per_pixel_resolver_rejects_invalid_camera_scale(
    camera_mm_per_pixel: object,
) -> None:
    with pytest.raises(ValueError, match="positive finite camera_mm_per_pixel"):
        mod.resolve_mm_per_pixel_for_coordinate_space(
            "source_camera_image_px",
            camera_mm_per_pixel=camera_mm_per_pixel,
            pixels_per_mm_projector=5.0,
        )
