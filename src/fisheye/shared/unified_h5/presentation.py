"""Citrus static presentation v1; preserve camera and raster transforms separately.

Grammar follows Citrus 0a9e2d4 src/logging/presentation_mapping_contract.cpp.
This verifies recorded logical mappings, not measured presentation timing.
"""

from .common import require, same_json
from .schema import read_json


def _frame(name, extent, description):
    return {
        "name": name,
        "extent_px": extent,
        "description": description,
        "origin": "top_left",
        "units": "px",
        "x_axis": "right",
        "y_axis": "down",
    }


def _mapping(
    source, destination, forward, inverse, *, note=None, height=None, reflected=False
):
    result = {
        "source_frame": source,
        "destination_frame": destination,
        "handedness": "reflected" if reflected else "preserved",
        "continuous_coordinate_mapping": {
            "coordinate_model": "continuous_pixel_edge_coordinates",
            "destination_from_source_homography_row_major": forward,
            "source_from_destination_homography_row_major": inverse,
        },
    }
    if note is not None:
        result["semantic_note"] = note
    if height is not None:
        result["continuous_coordinate_mapping"].update(
            {
                "height_px": height,
                "equations": {
                    "x_destination": "x_source",
                    "y_destination": "height_px - y_source",
                },
            }
        )
        result["discrete_pixel_index_mapping"] = {
            "coordinate_model": "integer_pixel_indices_at_pixel_centers",
            "height_px": height,
            "equations": {
                "column_destination": "column_source",
                "row_destination": "(height_px - 1) - row_source",
            },
        }
    return result


def validate_presentation(h5, receipt):
    a = h5["/geometry/correspondence/input"].attrs
    r = h5["/geometry/correspondence/input/renderer"].attrs
    width, height = int(a["final_display_width_px"]), int(a["final_display_height_px"])
    arena_width, arena_height = int(r["texture_width_px"]), int(r["texture_height_px"])
    x, y = float(a["effective_arena_origin_x_px"]), float(
        a["effective_arena_origin_y_px"]
    )
    arena, display, framebuffer, video = (
        "arena_relative_canvas_px",
        "final_display_canvas_px",
        "stimulus_display_framebuffer_px",
        "stimulus_video_px",
    )
    reflected_display = [1.0, 0.0, 0.0, 0.0, -1.0, float(height), 0.0, 0.0, 1.0]
    reflected_video = [1.0, 0.0, 0.0, 0.0, -1.0, float(arena_height), 0.0, 0.0, 1.0]
    expected = {
        "schema_id": "citrus.presentation_mapping",
        "schema_version": 1,
        "status": "complete",
        "arena_id": a["arena_id"],
        "coordinate_frames": {
            arena: _frame(
                arena,
                [arena_width, arena_height],
                "Logical scientific coordinates inside the active arena texture.",
            ),
            display: _frame(
                display,
                [width, height],
                "Logical full-canvas composition buffer before OpenGL presentation.",
            ),
            framebuffer: _frame(
                framebuffer,
                [width, height],
                "Top-left view of the raster delivered to the selected stimulus output.",
            ),
            video: _frame(
                video,
                [arena_width, arena_height],
                "Decoded top-left stimulus-video pixels after the encoder input reflection.",
            ),
        },
        "contract_semantics": {
            "camera_mapping_rule": "Use the accepted source_camera_px <-> final_display_canvas_px homography directly. Do not compose this presentation reflection into that camera homography.",
            "historical_compatibility": "Missing presentation metadata means legacy_unknown; it does not change the meaning of existing scientific arrays.",
            "mount_orientation_rule": "Fixed projector/display/optical orientation is calibration evidence captured by the accepted camera-to-logical-canvas homography, never a stimulus-local axis flip.",
            "scientific_coordinate_rule": "Scientific stimulus positions remain logical arena_relative_canvas_px with top-left origin, +X right, +Y down.",
        },
        "mappings": {
            "arena_relative_canvas_to_final_display_canvas": _mapping(
                arena,
                display,
                [1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0],
                [1.0, 0.0, -x, 0.0, 1.0, -y, 0.0, 0.0, 1.0],
            ),
            "arena_relative_canvas_to_stimulus_display_framebuffer": _mapping(
                arena,
                framebuffer,
                [1.0, 0.0, x, 0.0, -1.0, height - y, 0.0, 0.0, 1.0],
                [1.0, 0.0, -x, 0.0, -1.0, height - y, 0.0, 0.0, 1.0],
                reflected=True,
                note="Arena placement followed by the one presentation reflection.",
            ),
            "arena_relative_canvas_to_stimulus_video": _mapping(
                arena,
                video,
                reflected_video,
                reflected_video,
                reflected=True,
                height=arena_height,
                note="The encoded stimulus source is reflected vertically exactly once before overlays and encoding.",
            ),
            "final_display_canvas_to_stimulus_display_framebuffer": _mapping(
                display,
                framebuffer,
                reflected_display,
                reflected_display,
                reflected=True,
                height=height,
                note="The CUDA composition buffer is presented through OpenGL texture coordinates with one vertical reflection.",
            ),
        },
    }
    require(
        same_json(read_json(h5, receipt["presentation_ref"]), expected),
        "geometry_presentation_mapping_mismatch",
    )
    group, dataset = h5["/geometry/presentation"], h5[receipt["presentation_ref"]]
    # A reused current-format presentation carries these independent claims;
    # a native-only newly created presentation may have no such attributes.
    if "contract_sha256" in group.attrs or "checksum_sha256" in dataset.attrs:
        require(
            group.attrs.get("contract_sha256")
            == dataset.attrs.get("checksum_sha256")
            == receipt["presentation_sha256"]
            and group.attrs.get("status") == "complete",
            "geometry_presentation_digest_attributes",
        )
