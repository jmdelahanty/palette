from __future__ import annotations

import copy

import numpy as np
import pytest
import zarr

from fisheye.analysis import track_kinematics as mod
from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    CoordinateDescriptorError,
    build_coordinate_descriptor,
    load_coordinate_descriptor_attrs,
    stamp_coordinate_descriptor,
)


class _Node:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}


def _refined_positions_descriptor():
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
            "analysis/stimulus_runs/stimulus_a/arena_geometry"
        ),
        pixel_convention="continuous",
        row_identity_mode="frame_indices",
        row_identity_array_ref="camera_frame_ids",
        source_camera_overlay="requires_transform",
    )


def test_selected_positions_array_descriptor_overrides_parent_compatibility_attr() -> None:
    refined_group = _Node()
    refined_group.attrs["coordinate_space"] = "source_camera_image_px"
    positions_array = _Node()
    source_descriptor = _refined_positions_descriptor()
    stamp_coordinate_descriptor(positions_array, source_descriptor)

    selected = mod.load_positions_px_coordinate_descriptor(positions_array)

    assert refined_group.attrs["coordinate_space"] != selected.space_id
    assert selected.space_id == "arena_relative_canvas_px"
    assert mod.resolve_mm_per_pixel_for_coordinate_space(
        selected.space_id,
        camera_mm_per_pixel=9.0,
        pixels_per_mm_projector=4.0,
    ) == pytest.approx(0.25)


def test_selected_positions_array_rejects_missing_and_tampered_descriptor() -> None:
    positions_array = _Node()
    with pytest.raises(CoordinateDescriptorError) as missing_exc:
        mod.load_positions_px_coordinate_descriptor(positions_array)
    assert {issue.code for issue in missing_exc.value.issues} == {
        "descriptor_attr_missing",
        "descriptor_digest_missing",
    }

    stamp_coordinate_descriptor(positions_array, _refined_positions_descriptor())
    tampered = copy.deepcopy(positions_array.attrs[COORDINATE_DESCRIPTOR_ATTR])
    tampered["reference_extent"]["width"] = 4512
    positions_array.attrs[COORDINATE_DESCRIPTOR_ATTR] = tampered

    with pytest.raises(CoordinateDescriptorError) as tampered_exc:
        mod.load_positions_px_coordinate_descriptor(positions_array)
    assert {issue.code for issue in tampered_exc.value.issues} == {
        "descriptor_digest_mismatch"
    }


def test_writer_stamps_each_track_positions_array_with_rebound_row_identity() -> None:
    tracks, summaries = mod.build_track_datasets(
        track_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        frames=np.array([10, 11, 20, 21], dtype=np.int64),
        positions_px=np.array(
            [[0.0, 0.0], [1.0, 0.0], [4.0, 5.0], [5.0, 5.0]],
            dtype=np.float32,
        ),
        headings_deg=np.zeros(4, dtype=np.float32),
        keypoint_success=np.ones(4, dtype=bool),
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=0.25,
    )
    run_group = zarr.open_group("memory://track-coordinate-contract", mode="w")
    source_descriptor = _refined_positions_descriptor()

    ordered_ids = mod.save_track_kinematics_tracks(
        run_group,
        tracks,
        summaries,
        positions_px_descriptor=source_descriptor,
    )

    assert ordered_ids == [0, 1]
    assert source_descriptor.row_identity.array_ref == "camera_frame_ids"
    for track_id in ordered_ids:
        track_group = run_group["tracks"][f"id_{track_id}"]
        persisted = load_coordinate_descriptor_attrs(
            track_group["positions_px"].attrs
        )
        assert persisted.space_id == source_descriptor.space_id
        assert persisted.reference_extent == source_descriptor.reference_extent
        assert persisted.row_identity.mode == "track_frame_indices"
        assert persisted.row_identity.array_ref == "../frame_indices"
        assert COORDINATE_DESCRIPTOR_ATTR not in track_group["positions_mm"].attrs


def test_refined_source_descriptor_path_and_digest_are_normalized_in_provenance() -> None:
    descriptor = _refined_positions_descriptor()
    source_path = (
        "refined_online_runs/refined_a/interpolated/positions_px"
    )

    attrs = mod._track_kinematics_contract_attrs(
        run_type="online",
        method="track_kinematics_online_refined",
        parameters={"coordinate_space": descriptor.space_id},
        inputs={
            "refined_online_run": "refined_a",
            "stimulus_run": "stimulus_a",
            "chaser_index": 0,
            "positions_px_source_path": source_path,
            "positions_px_coordinate_descriptor_sha256": descriptor.digest(),
        },
    )

    assert attrs["source_refs"]["source_positions_px_path"] == source_path
    assert (
        attrs["source_refs"][
            "source_positions_px_coordinate_descriptor_sha256"
        ]
        == descriptor.digest()
    )
