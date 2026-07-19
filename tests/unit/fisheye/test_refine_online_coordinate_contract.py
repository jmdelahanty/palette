from __future__ import annotations

from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console
import zarr
from zarr.storage import MemoryStore

from fisheye.refinement import refine_online_detect as mod
from fisheye.shared.coordinate_descriptor import (
    CoordinateDescriptorError,
    build_coordinate_descriptor,
    coordinate_descriptor_attrs,
    load_coordinate_descriptor_attrs,
)


SOURCE_PATH = "analysis/stimulus_runs/stim_1/tracking_data/chaser_states"
STIMULUS_PATH = "analysis/stimulus_runs/stim_1"


def _arena_source_attrs() -> dict[str, object]:
    return {
        "coordinate_frame": "arena_relative_canvas_px",
        "coordinate_units": "px",
        "coordinate_origin": "top_left_of_active_arena",
        "position_fields": (
            "chaser_pos_x,chaser_pos_y,target_pos_x,target_pos_y,"
            "target_clamped_pos_x,target_clamped_pos_y"
        ),
        "x_axis_direction": "right",
        "y_axis_direction": "down",
    }


def _texture_source_attrs() -> dict[str, object]:
    attrs = _arena_source_attrs()
    attrs.update(
        {
            "coordinate_frame": "texture",
            "coordinate_origin": "top_left_of_texture",
        }
    )
    return attrs


def _source_metadata(attrs: dict[str, object]) -> dict[str, object]:
    return {"source_path": SOURCE_PATH, "source_attrs": attrs}


def _canonical_source_descriptor():
    return build_coordinate_descriptor(
        space_id="arena_relative_canvas_px",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        origin="arena_top_left",
        positive_x="right",
        positive_y="down",
        reference_width=720,
        reference_height=640,
        reference_units="px",
        reference_authority=(
            f"{STIMULUS_PATH}/calibration/arena_geometry.attrs"
            "[arena_region_width_px,arena_region_height_px]"
        ),
        pixel_convention="continuous",
        row_identity_mode="sample_indices",
        row_identity_array_ref=f"{SOURCE_PATH}#rows",
        source_camera_overlay="requires_transform",
    )


def _resolve(
    attrs: dict[str, object],
    *,
    run_attrs: dict[str, object] | None = None,
    arena_attrs: dict[str, object] | None = None,
):
    return mod.resolve_online_coordinate_descriptor(
        _source_metadata(attrs),
        stimulus_run_path=STIMULUS_PATH,
        stimulus_run_attrs=run_attrs or {},
        arena_geometry_attrs=arena_attrs,
    )


def test_resolver_accepts_an_intact_canonical_descriptor() -> None:
    descriptor = _canonical_source_descriptor()
    attrs = coordinate_descriptor_attrs(descriptor)

    resolved = _resolve(attrs)

    assert resolved == descriptor

    attrs["coordinate_descriptor"]["reference_extent"]["width"] = 999
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(attrs)
    assert {issue.code for issue in exc_info.value.issues} == {
        "descriptor_digest_mismatch"
    }


def test_resolver_rejects_valid_nonpixel_descriptor_for_positions_px() -> None:
    descriptor = build_coordinate_descriptor(
        space_id="detector_normalized_xy",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("normalized", "normalized"),
        origin="top_left",
        positive_x="right",
        positive_y="down",
        reference_width=640,
        reference_height=640,
        reference_units="px",
        reference_authority="/raw_video/images_ds.shape[-2:]",
        pixel_convention="continuous",
        row_identity_mode="sample_indices",
        row_identity_array_ref=f"{SOURCE_PATH}#rows",
        source_camera_overlay="requires_transform",
    )

    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(coordinate_descriptor_attrs(descriptor))
    assert {issue.code for issue in exc_info.value.issues} == {
        "online_pixel_space_required"
    }


def test_resolver_maps_arena_relative_only_from_complete_source_and_extent() -> None:
    descriptor = _resolve(
        _arena_source_attrs(),
        arena_attrs={
            "arena_region_width_px": 720,
            "arena_region_height_px": 640,
        },
    )

    assert descriptor.space_id == "arena_relative_canvas_px"
    assert descriptor.reference_extent.width == 720
    assert descriptor.reference_extent.height == 640
    assert descriptor.reference_extent.authority.startswith(
        f"{STIMULUS_PATH}/calibration/arena_geometry.attrs"
    )
    assert descriptor.legacy_space_label is None
    assert [ref.ref for ref in descriptor.lineage_refs] == [SOURCE_PATH]


@pytest.mark.parametrize(
    ("missing_key", "expected_code"),
    (
        ("coordinate_units", "online_source_units_missing"),
        ("coordinate_origin", "online_source_coordinate_attr_missing"),
        ("x_axis_direction", "online_source_coordinate_attr_missing"),
        ("y_axis_direction", "online_source_coordinate_attr_missing"),
        ("position_fields", "online_source_position_fields_missing"),
    ),
)
def test_arena_relative_legacy_metadata_fails_closed_when_incomplete(
    missing_key: str,
    expected_code: str,
) -> None:
    attrs = _arena_source_attrs()
    del attrs[missing_key]

    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(
            attrs,
            arena_attrs={
                "arena_region_width_px": 720,
                "arena_region_height_px": 640,
            },
        )
    assert expected_code in {issue.code for issue in exc_info.value.issues}


def test_arena_relative_requires_selected_run_arena_dimensions() -> None:
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(_arena_source_attrs(), arena_attrs=None)
    assert {issue.code for issue in exc_info.value.issues} == {
        "online_arena_geometry_missing"
    }

    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(
            _arena_source_attrs(),
            arena_attrs={"arena_region_width_px": 720},
        )
    assert "online_reference_extent_invalid" in {
        issue.code for issue in exc_info.value.issues
    }


def test_explicit_legacy_texture_requires_exact_consistent_transform_evidence() -> None:
    transform = {
        "scope": "run_level_legacy_texture_space",
        "texture_dimensions": [358, 358],
        "camera_dimensions": [4512, 4512],
        "texture_to_camera_scale": 4512 / 358,
    }
    run_attrs = {
        "coordinate_transform_status": "legacy_run_level_texture_to_camera",
        "coordinate_transform": json.dumps(transform, sort_keys=True),
    }

    descriptor = _resolve(
        _texture_source_attrs(),
        run_attrs=run_attrs,
    )

    assert descriptor.space_id == "stimulus_texture_px"
    assert descriptor.legacy_space_label == "texture"
    assert descriptor.reference_extent.width == 358
    assert descriptor.reference_extent.height == 358
    assert len(descriptor.transform_refs) == 1
    assert descriptor.transform_refs[0].sha256 is not None

    bad_transform = dict(transform)
    bad_transform["camera_dimensions"] = [4512, 4000]
    run_attrs["coordinate_transform"] = bad_transform
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(_texture_source_attrs(), run_attrs=run_attrs)
    assert "online_legacy_texture_scale_inconsistent" in {
        issue.code for issue in exc_info.value.issues
    }


def test_explicit_legacy_texture_without_evidence_fails_closed() -> None:
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        _resolve(_texture_source_attrs())
    assert {issue.code for issue in exc_info.value.issues} == {
        "online_legacy_texture_evidence_missing"
    }


def test_load_online_positions_keeps_tuple_shape_and_exposes_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=3)
    stimulus = root.require_group("analysis").require_group("stimulus_runs").create_group("stim_1")
    arena = stimulus.require_group("calibration").create_group("arena_geometry")
    arena.attrs.update(
        {
            "arena_region_width_px": 720,
            "arena_region_height_px": 640,
        }
    )
    bundle = SimpleNamespace(
        online={
            "target_pos_x": np.asarray([10.0, 11.0]),
            "target_pos_y": np.asarray([20.0, 21.0]),
        },
        camera_frame_ids=np.asarray([100, 101], dtype=np.int64),
        provenance={"stimulus_run": "stim_1"},
        online_coordinate_metadata=_source_metadata(_arena_source_attrs()),
    )
    monkeypatch.setattr(mod, "load_chaser_metrics", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(mod.zarr, "open", lambda *args, **kwargs: root)
    monkeypatch.setattr(
        mod,
        "load_run_calibration",
        lambda *args, **kwargs: SimpleNamespace(
            texture_to_camera_scale=1.0,
            pixels_per_mm_projector=None,
            source="test",
        ),
    )

    result = mod.load_online_positions(
        "unused.zarr",
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert len(result) == 6
    frames, positions, valid, scale, pixels_per_mm, metadata = result
    np.testing.assert_array_equal(frames, [100, 101])
    np.testing.assert_allclose(positions, [[10.0, 20.0], [11.0, 21.0]])
    np.testing.assert_array_equal(valid, [True, True])
    assert scale == 1.0
    assert pixels_per_mm is None
    descriptor = load_coordinate_descriptor_attrs(
        {
            "coordinate_descriptor": metadata["coordinate_descriptor"],
            "coordinate_descriptor_sha256": metadata[
                "coordinate_descriptor_sha256"
            ],
        }
    )
    assert descriptor.space_id == "arena_relative_canvas_px"
    assert descriptor.row_identity.array_ref == "camera_frame_ids"


def test_refined_outputs_preserve_native_descriptor_and_bind_output_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=3)
    descriptor = _canonical_source_descriptor()
    source_metadata = _source_metadata(_arena_source_attrs())
    frames = np.arange(100, 105, dtype=np.int64)
    positions = np.column_stack(
        [np.linspace(10.0, 14.0, 5), np.linspace(20.0, 24.0, 5)]
    )
    metadata = {
        "stimulus_run": "stim_1",
        "chaser_index": 0,
        "total_frames": 5,
        "valid_frames": 5,
        "coverage_percent": 100.0,
        "texture_to_camera_scale": 1.0,
        "pixels_per_mm_projector": None,
        "coordinate_space": descriptor.space_id,
        "coordinate_descriptor": descriptor.to_dict(),
        "coordinate_descriptor_sha256": descriptor.digest(),
        "online_coordinate_source": source_metadata,
    }
    monkeypatch.setattr(
        mod,
        "load_online_positions",
        lambda *args, **kwargs: (
            frames,
            positions,
            np.ones(5, dtype=bool),
            1.0,
            None,
            metadata,
        ),
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *args, **kwargs: root)
    monkeypatch.setattr(mod, "get_git_info", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda *args, **kwargs: {"platform": {}},
    )

    run_name = mod.refine_online_positions(
        "unused.zarr",
        window_length=3,
        polyorder=1,
        displacement_threshold=1000.0,
        max_gap=2,
        console=Console(file=StringIO(), force_terminal=False),
        created_at_utc="2026-07-18T12:00:00+00:00",
    )

    run = root[mod.REFINED_ONLINE_GROUP][run_name]
    assert run.attrs["coordinate_space"] == "arena_relative_canvas_px"
    assert "legacy_space_label" not in run.attrs
    assert run.attrs["positions_coordinate_descriptor_refs"] == [
        "filtered/positions_px",
        "interpolated/positions_px",
    ]
    for subgroup_name in ("filtered", "interpolated"):
        subgroup = run[subgroup_name]
        output_descriptor = load_coordinate_descriptor_attrs(
            subgroup["positions_px"].attrs
        )
        assert output_descriptor.space_id == "arena_relative_canvas_px"
        assert output_descriptor.row_identity.mode == "frame_indices"
        assert output_descriptor.row_identity.array_ref == "camera_frame_ids"
        assert SOURCE_PATH in {ref.ref for ref in output_descriptor.lineage_refs}
        np.testing.assert_array_equal(subgroup["camera_frame_ids"][:], frames)
