from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

from fisheye.visualization import visualize_online_tracking as mod


class _Handoff:
    def __init__(self, *, space_id: str = "arena_relative_canvas_px") -> None:
        self.checks = 0
        self.coordinate_descriptor = SimpleNamespace(
            descriptor=SimpleNamespace(
                profile_id="arena_relative_canvas_px.top_left_y_down.v1",
                space_id=space_id,
                geometry_type="point_xy",
                components=("x", "y"),
                component_units=("px", "px"),
                reference_extent=SimpleNamespace(
                    width=320,
                    height=240,
                    units="px",
                ),
                source_camera_overlay=SimpleNamespace(
                    status="requires_transform",
                    transform_refs=(
                        SimpleNamespace(
                            record_ref="/transform@record",
                            record_sha256="a" * 64,
                        ),
                    ),
                ),
            )
        )

    def assert_verified(self) -> None:
        self.checks += 1


def _bundle(*, handoff: _Handoff | None = None, timestamps: bool = True):
    return SimpleNamespace(
        online_coordinate_handoff=handoff or _Handoff(),
        online={
            "target_position_xy": np.asarray(
                [[10.0, 20.0], [11.0, 20.0], [np.nan, np.nan], [13.0, 21.0]],
                dtype=np.float64,
            )
        },
        camera_frame_ids=np.asarray([100, 101, 102, 103], dtype=np.int64),
        timestamp_ns=(
            np.asarray([1_000, 2_000, 3_000, 4_000], dtype=np.int64)
            if timestamps
            else np.full(4, -1, dtype=np.int64)
        ),
        provenance={"stimulus_run": "stimulus_1"},
    )


def test_verified_surface_keeps_native_arena_coordinates_and_extent() -> None:
    bundle = _bundle()

    surface = mod._verified_online_tracking_surface(bundle)

    np.testing.assert_allclose(
        surface.positions,
        bundle.online["target_position_xy"],
        equal_nan=True,
    )
    assert surface.space_id == "arena_relative_canvas_px"
    assert (surface.width_px, surface.height_px) == (320, 240)
    assert bundle.online_coordinate_handoff.checks == 2


def test_verified_surface_rejects_unsupported_or_untyped_coordinates() -> None:
    bundle = _bundle()
    bundle.online_coordinate_handoff = None

    with pytest.raises(ValueError, match="canonical typed coordinate handoff"):
        mod._verified_online_tracking_surface(bundle)


def test_verified_surface_rejects_coordinate_name_guess() -> None:
    bundle = _bundle(handoff=_Handoff(space_id="source_camera_image_px"))

    with pytest.raises(ValueError, match="supported canonical arena-relative"):
        mod._verified_online_tracking_surface(bundle)


def test_visualization_uses_persisted_timestamps_and_never_invents_mm(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    monkeypatch.setattr(mod, "load_chaser_metrics", lambda *_args, **_kwargs: bundle)
    output = tmp_path / "online.png"
    console = Console(record=True)

    mod.visualize_online_tracking(
        "unused.zarr",
        output_path=str(output),
        console=console,
    )

    assert output.is_file()
    rendered = console.export_text()
    assert "arena_relative_canvas_px (320x240px)" in rendered
    assert "Total distance: 1.0 px" in rendered
    assert " mm" not in rendered
    assert bundle.online_coordinate_handoff.checks == 3


def test_plot_axis_falls_back_to_camera_identity_without_timestamps() -> None:
    surface = mod._verified_online_tracking_surface(_bundle(timestamps=False))

    values, label, duration = mod._plot_axis(surface)

    np.testing.assert_array_equal(values, [100.0, 101.0, 102.0, 103.0])
    assert label == "Camera frame ID"
    assert duration is None
