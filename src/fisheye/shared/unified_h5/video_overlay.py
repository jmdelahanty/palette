"""Video tracking overlay states (admission revision 7, video_overlay_v1).

``/components/video_tracking_overlay/states`` records, per stimulus update,
the fish dot and heading arrow the encoder draws on the session video (never
projected), and ``settings_json`` the session settings. The contract is
agent-contracts ``citrus-palette/video_tracking_overlay/
video_tracking_overlay_states_v1.json`` (vendored as
``contracts/video_tracking_overlay_states_v1.json``); the catalog validates
layout, this module validates its invariants against ``/frames/stimulus``,
``/observations/bounding_boxes``, ``/observations/pose`` and the captured
geometry, and re-derives the dot position, arrow origin and arrow direction
exactly as Citrus ``src/core/tracking_target_view.cpp`` (1855e374) computes
them: double projection with a true division, binary32 casts, the OpenCV 4.10
float ``boundingRect`` (floor of min and max, width = floor(max) - floor(min)
+ 1), binary32 shifts and centres. Nothing is toleranced but the stored unit
length.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import h5py
import numpy as np

from .common import require
from .geometry import NAMESPACE_BINDINGS
from .hdf5_types import iter_blocks
from .pose import BINDING
from .pose import CONTRACT as POSE_CONTRACT
from .pose import GROUP as POSE_GROUP
from .pose import _contract_version as pose_contract_version
from .pose import _json as read_small_json
from .pose_v3 import DEGENERATE_EPSILON

COMPONENT = "video_tracking_overlay"
STATES = "/components/video_tracking_overlay/states"
SETTINGS = "/components/video_tracking_overlay/settings_json"
FRAMES = "/frames/stimulus"
BOXES = "/observations/bounding_boxes"
RUNTIME_GEOMETRY = "/runtime_geometry_contract/contract_json"
CALIBRATION = "/calibration_snapshot"
INPUT_NONE, INPUT_CURRENT, INPUT_HELD = 0, 1, 2
HEADING_VALID = 7
PROBE_LENGTH_PX = np.float32(20.0)
PROJECTION_EPSILON = 1e-12
UNIT_TOLERANCE = 1e-6
MAX_THICKNESS_PX = 64
SETTINGS_KEYS = frozenset(
    {
        "schema_id", "schema_version", "enabled", "radius_mm", "radius_px",
        "pixels_per_mm_projector", "scale_authority", "color_authored_rgba",
        "color_quantised_rgba", "color_quantiser", "held_policy", "position_rule",
        "position_frame", "video_flip_rule", "source", "pose_observation_enabled",
        "heading_arrow", "settings_load_status",
    }
)
ARROW_KEYS = frozenset(
    {
        "enabled", "active", "length_mm", "length_px", "color_authored_rgba",
        "color_quantised_rgba", "thickness_px", "origin_rule", "direction_rule",
        "probe_length_camera_px", "heading_gate", "pose_source",
    }
)
SETTINGS_CONSTANTS = {
    "schema_id": "citrus.video_tracking_overlay",
    "schema_version": 1,
    "scale_authority": "calibration_snapshot.camera.pixels_per_mm_projector",
    "color_quantiser": "rgba8_round_half_up_v1",
    "position_rule": "projected_bbox_rect_center_v1",
    "position_frame": "arena_local_px",
    "video_flip_rule": "video_y = height - y",
    "source": "tracking_target_view",
    "settings_load_status": "loaded",
}
ARROW_CONSTANTS = {
    "origin_rule": "projected_pose_heading_origin_v1",
    "direction_rule": "two_probe_canvas_midline_v1",
    "probe_length_camera_px": 20.0,
    "heading_gate": "pose_contract_minimum_keypoint_confidence_or_heading_split_step_gate",
    "pose_source": "current_pose_only",
}
# Every field with explicit validity holds payload zero when its flag is 0.
VALIDITY = {
    "video_frame_index_valid": ("video_frame_index",),
    "position_valid": ("position_arena_local_px",),
    "source_camera_id_valid": ("source_camera_id",),
    "box_sequence_id_valid": ("box_sequence_id",),
    "box_recording_frame_id_valid": ("box_recording_frame_id",),
    "heading_valid": ("heading_camera_angle_degrees", "heading_origin_camera_px"),
    "midline_valid": ("arrow_origin_arena_local_px", "arrow_direction_canvas"),
    "pose_sequence_id_valid": ("pose_sequence_id",),
    "pose_recording_frame_id_valid": ("pose_recording_frame_id",),
}
FLAGS = (
    "video_frame_index_valid", "decision_draw", "drawn", "position_valid",
    "source_camera_id_valid", "box_sequence_id_valid", "box_recording_frame_id_valid",
    "heading_valid", "midline_valid", "arrow_drawn", "pose_sequence_id_valid",
    "pose_recording_frame_id_valid",
)
F32 = np.float32


# --- exact arithmetic (tracking_target_view.cpp) ------------------------------


def project(h: np.ndarray, x, y):
    """ProjectCameraPoint: binary32 in, double projection, binary32 out.

    Returns ``(ok, x', y')``; ``ok`` is false where ``|w| <= 1e-12``.
    """

    xd = np.asarray(x, dtype=np.float32).astype(np.float64)
    yd = np.asarray(y, dtype=np.float32).astype(np.float64)
    w = h[2, 0] * xd + h[2, 1] * yd + h[2, 2]
    ok = np.abs(w) > PROJECTION_EPSILON
    with np.errstate(all="ignore"):
        safe = np.where(ok, w, 1.0)
        px = ((h[0, 0] * xd + h[0, 1] * yd + h[0, 2]) / safe).astype(np.float32)
        py = ((h[1, 0] * xd + h[1, 1] * yd + h[1, 2]) / safe).astype(np.float32)
    return ok, px, py


def box_arena_local(h: np.ndarray, origin: tuple[int, int], x_min, y_min, width, height):
    """ProjectBoxToArenaLocal: rect centre of the projected corners, arena-local."""

    x_min, y_min = np.asarray(x_min, np.float32), np.asarray(y_min, np.float32)
    x_max = x_min + np.asarray(width, np.float32)
    y_max = y_min + np.asarray(height, np.float32)
    corners = [project(h, cx, cy) for cx, cy in ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))]
    ok = np.logical_and.reduce([c[0] for c in corners])
    xs = np.stack([c[1] for c in corners])
    ys = np.stack([c[2] for c in corners])
    with np.errstate(all="ignore"):
        finite = np.isfinite(xs).all(axis=0) & np.isfinite(ys).all(axis=0)
        ok &= finite
        safe_x, safe_y = np.where(ok, xs, 0), np.where(ok, ys, 0)
        left = np.floor(safe_x.min(axis=0)).astype(np.int64)
        right = np.floor(safe_x.max(axis=0)).astype(np.int64)
        top = np.floor(safe_y.min(axis=0)).astype(np.int64)
        bottom = np.floor(safe_y.max(axis=0)).astype(np.int64)
    rect_x = left.astype(np.float32) - F32(origin[0])
    rect_y = top.astype(np.float32) - F32(origin[1])
    rect_w = (right - left + 1).astype(np.float32)
    rect_h = (bottom - top + 1).astype(np.float32)
    centre_x = rect_x + rect_w / F32(2.0)
    centre_y = rect_y + rect_h / F32(2.0)
    return ok, centre_x, centre_y


def _unit(from_x, from_y, to_x, to_y):
    dx = to_x - from_x
    dy = to_y - from_y
    dx64, dy64 = dx.astype(np.float64), dy.astype(np.float64)
    with np.errstate(all="ignore"):
        length = np.sqrt(dx64 * dx64 + dy64 * dy64)
        ok = np.isfinite(length) & (length > DEGENERATE_EPSILON)
        safe = np.where(ok, length, 1.0)
        return ok, (dx64 / safe).astype(np.float32), (dy64 / safe).astype(np.float32)


def midline(h: np.ndarray, origin: tuple[int, int], origin_x, origin_y, direction_x, direction_y):
    """ProjectMidline: arrow origin (arena-local) and unit canvas direction."""

    ox, oy = np.asarray(origin_x, np.float32), np.asarray(origin_y, np.float32)
    dx, dy = np.asarray(direction_x, np.float32), np.asarray(direction_y, np.float32)
    left_x, left_y = dy, -dx
    with np.errstate(all="ignore"):
        finite = np.isfinite(ox) & np.isfinite(oy) & np.isfinite(dx) & np.isfinite(dy)
        ok0, cx, cy = project(h, ox, oy)
        ok1, fx, fy = project(h, ox + PROBE_LENGTH_PX * dx, oy + PROBE_LENGTH_PX * dy)
        ok2, lx, ly = project(h, ox + PROBE_LENGTH_PX * left_x, oy + PROBE_LENGTH_PX * left_y)
        ok3, ux, uy = _unit(cx, cy, fx, fy)
        ok4, _, _ = _unit(cx, cy, lx, ly)
    ok = finite & ok0 & ok1 & ok2 & ok3 & ok4
    arena_x = cx - F32(origin[0])
    arena_y = cy - F32(origin[1])
    return ok, arena_x, arena_y, ux, uy


def quantise_rgba8(rgba) -> list[int]:
    """CitrusColor::QuantizeRoundHalfUpV1 (lround of double(float) * 255, clamped)."""

    result = []
    for value in rgba:
        x = float(np.float32(value))
        if not math.isfinite(x):
            result.append(0)
            continue
        scaled = x * 255.0
        rounded = math.floor(abs(scaled) + 0.5) * (1 if scaled >= 0 else -1)
        result.append(int(min(max(rounded, 0), 255)))
    return result


# --- inputs -------------------------------------------------------------------


def _resolve(path: str) -> str:
    for logical, physical in NAMESPACE_BINDINGS.items():
        if path == logical or path.startswith(logical + "/"):
            return physical + path[len(logical):]
    return path


def _homography(h5, camera_serial: str) -> np.ndarray:
    path = _resolve(f"{CALIBRATION}/{camera_serial}/homography_matrix")
    require(path in h5 and isinstance(h5[path], h5py.Dataset), "video_overlay_homography_missing")
    dataset = h5[path]
    require(dataset.shape == (3, 3) and dataset.dtype == np.dtype("<f8"), "video_overlay_homography_type")
    values = dataset[()]
    require(bool(np.isfinite(values).all()), "video_overlay_homography_nonfinite")
    return values


def _arena_origin(h5) -> tuple[int, int]:
    """The integer arena origin the view subtracts (effective geometry)."""

    path = _resolve(RUNTIME_GEOMETRY)
    require(path in h5, "video_overlay_runtime_geometry_missing")
    contract = read_small_json(h5, path)
    origin = (
        contract.get("effective_geometry", {}).get("arena_region", {}).get("origin_px")
        if isinstance(contract.get("effective_geometry"), Mapping)
        else None
    )
    require(isinstance(origin, Mapping), "video_overlay_arena_origin_missing")
    values = []
    for axis in ("x", "y"):
        value = origin.get(axis)
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and float(value).is_integer(),
            "video_overlay_arena_origin_not_integer",
        )
        values.append(int(value))
    return values[0], values[1]


def _projector_scale(h5, camera_serial: str) -> float:
    path = _resolve(f"{CALIBRATION}/{camera_serial}")
    require(path in h5 and "pixels_per_mm_projector" in h5[path].attrs, "video_overlay_scale_authority_missing")
    value = h5[path].attrs["pixels_per_mm_projector"]
    require(np.asarray(value).shape == () and np.isfinite(value) and float(value) > 0, "video_overlay_scale_authority_invalid")
    return float(np.float32(value))


def _number(value: object, reason: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), reason)
    return float(value)


def _f32_product(a: float, b: float) -> float:
    return float(F32(a) * F32(b))


def validate_settings(h5, settings: Mapping, *, camera_serial: str, pose_enabled: bool) -> dict:
    """The session settings JSON against the contract and the scale authority."""

    require(set(settings) == SETTINGS_KEYS, "video_overlay_settings_keys")
    for key, value in SETTINGS_CONSTANTS.items():
        require(settings[key] == value and type(settings[key]) is type(value), f"video_overlay_settings_constant:{key}")
    require(settings["enabled"] is True, "video_overlay_settings_not_enabled")
    require(settings["held_policy"] in ("draw", "skip"), "video_overlay_settings_held_policy")
    require(type(settings["pose_observation_enabled"]) is bool, "video_overlay_settings_pose_flag")
    require(settings["pose_observation_enabled"] is pose_enabled, "video_overlay_settings_pose_flag_mismatch")
    scale = _projector_scale(h5, camera_serial)
    recorded = _number(settings["pixels_per_mm_projector"], "video_overlay_settings_scale")
    require(recorded == scale, "video_overlay_scale_authority_mismatch")
    radius_mm = _number(settings["radius_mm"], "video_overlay_settings_radius")
    require(radius_mm > 0 and float(F32(radius_mm)) == radius_mm, "video_overlay_settings_radius")
    require(_number(settings["radius_px"], "video_overlay_settings_radius_px") == _f32_product(radius_mm, scale), "video_overlay_settings_radius_px")
    _colour(settings["color_authored_rgba"], settings["color_quantised_rgba"], "dot")
    arrow = settings["heading_arrow"]
    require(isinstance(arrow, Mapping) and set(arrow) == ARROW_KEYS, "video_overlay_settings_arrow_keys")
    for key, value in ARROW_CONSTANTS.items():
        require(arrow[key] == value and type(arrow[key]) is type(value), f"video_overlay_settings_arrow_constant:{key}")
    require(type(arrow["enabled"]) is bool and type(arrow["active"]) is bool, "video_overlay_settings_arrow_flags")
    require(arrow["active"] is (arrow["enabled"] and pose_enabled), "video_overlay_settings_arrow_active")
    length_mm = _number(arrow["length_mm"], "video_overlay_settings_arrow_length")
    require(length_mm > 0 and float(F32(length_mm)) == length_mm, "video_overlay_settings_arrow_length")
    require(_number(arrow["length_px"], "video_overlay_settings_arrow_length_px") == _f32_product(length_mm, scale), "video_overlay_settings_arrow_length_px")
    require(
        type(arrow["thickness_px"]) is int and 1 <= arrow["thickness_px"] <= MAX_THICKNESS_PX,
        "video_overlay_settings_arrow_thickness",
    )
    _colour(arrow["color_authored_rgba"], arrow["color_quantised_rgba"], "arrow")
    return {"held_policy": settings["held_policy"], "arrow_active": arrow["active"]}


def _colour(authored: object, quantised: object, name: str) -> None:
    require(
        isinstance(authored, list) and len(authored) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in authored),
        f"video_overlay_settings_colour:{name}",
    )
    require(
        isinstance(quantised, list) and len(quantised) == 4
        and all(type(v) is int for v in quantised)
        and quantised == quantise_rgba8(authored),
        f"video_overlay_settings_colour_quantised:{name}",
    )


def _payload_zero(rows, name: str) -> np.ndarray:
    values = rows[name] == 0
    if values.ndim == 1:
        return values
    return values.all(axis=tuple(range(1, values.ndim)))


def _bits(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


# --- joins ----------------------------------------------------------------------


def _frame_index(h5) -> dict[int, tuple[int, int]]:
    """stimulus_frame_num -> (video_frame_index, video_frame_index_valid)."""

    frames = {}
    for _, block in iter_blocks(h5[FRAMES]):
        for frame, index, valid in zip(
            block["stimulus_frame_num"].tolist(),
            block["video_frame_index"].tolist(),
            block["video_frame_index_valid"].tolist(),
        ):
            frames[int(frame)] = (int(index), int(valid))
    return frames


def _box_rows(h5, wanted: set[tuple[int, int, int]]) -> dict[tuple[int, int, int], list[tuple]]:
    """(camera, recording frame, box index) -> [(payload time, x, y, w, h)]."""

    found: dict[tuple[int, int, int], list[tuple]] = {key: [] for key in wanted}
    if not wanted or BOXES not in h5:
        return found
    frames = np.fromiter({key[1] for key in wanted}, dtype=np.uint64)
    for _, block in iter_blocks(h5[BOXES]):
        for offset in np.flatnonzero(np.isin(block["recording_frame_id"], frames)):
            row = block[offset]
            key = (int(row["payload_camera_id"]), int(row["recording_frame_id"]), int(row["box_index_in_payload"]))
            if key in found:
                found[key].append(
                    (
                        int(row["payload_timestamp_ns_monotonic"]),
                        row["x_min"], row["y_min"], row["width"], row["height"],
                    )
                )
    return found


def _pose_objects(h5, wanted: set[int]) -> dict[int, dict]:
    """pose sequence_id -> {camera, recording_frame_id, objects by payload index}."""

    result: dict[int, dict] = {}
    if not wanted:
        return result
    updates, objects = h5[POSE_GROUP + "/updates"], h5[POSE_GROUP + "/objects"]
    rows: dict[int, int] = {}
    for start, block in iter_blocks(updates):
        for offset in np.flatnonzero(np.isin(block["sequence_id"], list(wanted))):
            sequence = int(block["sequence_id"][offset])
            require(sequence not in result, "video_overlay_pose_sequence_duplicate")
            rows[start + int(offset)] = sequence
            result[sequence] = {
                "camera_id": int(block["camera_id"][offset]),
                "recording_frame_id": int(block["recording_frame_id"][offset]),
                "objects": {},
            }
    for _, block in iter_blocks(objects):
        for offset in np.flatnonzero(np.isin(block["update_row"], list(rows))):
            row = block[offset]
            entry = result[rows[int(row["update_row"])]]["objects"]
            index = int(row["payload_object_index"])
            require(index not in entry, "video_overlay_pose_object_duplicate")
            entry[index] = row.copy()
    return result


# --- the validator ----------------------------------------------------------------


def declared(h5, outcomes: Mapping | None = None) -> bool:
    present = (STATES in h5, SETTINGS in h5)
    if outcomes is not None:
        present += (COMPONENT in outcomes,)
    require(all(present) or not any(present), "video_overlay_declaration_incomplete")
    return all(present)


def validate_video_overlay_states(
    h5, binding: Mapping, *, camera_serial: str, outcomes: Mapping | None = None
) -> dict:
    """Validate the overlay states and settings; return counts for the receipt.

    A file whose binding does not admit the overlay (revisions 1 to 6) must
    not declare it. Rows whose bounding box was replaced by a later update of
    the same recording frame are accepted (the per-frame filter keeps the last
    update) and counted as ``superseded_box_rows``; their position is not
    recomputed because the box the overlay saw is not in the file.
    """

    if not declared(h5, outcomes):
        return {"rows": 0, "superseded_box_rows": 0}
    require(binding.get("admits_video_overlay") is True, "video_overlay_not_admitted_by_revision")
    pose_present = POSE_GROUP in h5
    if pose_present:
        require(pose_contract_version(h5) == 3, "pose_v3_required_by_video_overlay_v1")
    settings_json = read_small_json(h5, SETTINGS)
    pose_enabled = settings_json.get("pose_observation_enabled")
    require(pose_enabled is pose_present, "video_overlay_pose_presence_mismatch")
    settings = validate_settings(h5, settings_json, camera_serial=camera_serial, pose_enabled=pose_present)
    shaman_camera = read_small_json(h5, BINDING).get("shaman_numeric_camera_id")
    homography = _homography(h5, camera_serial)
    origin = _arena_origin(h5)
    frames = _frame_index(h5)

    states = h5[STATES]
    require(states.shape[0] == len(frames), f"video_overlay_coverage_gap:{states.shape[0]}!={len(frames)}")

    total, superseded = 0, 0
    last_frame, last_box, last_pose = -1, -1, -1
    for _start, rows in iter_blocks(states):
        n = len(rows)
        frame = rows["stimulus_frame_num"].astype(np.int64)
        require(
            n == 0 or (frame[0] > last_frame and bool(np.all(np.diff(frame) > 0))),
            "video_overlay_frame_order",
        )
        if n:
            last_frame = int(frame[-1])
        total += n
        for flag in FLAGS:
            require(bool(np.all((rows[flag] == 0) | (rows[flag] == 1))), f"video_overlay_flag_not_boolean:{flag}")
        for flag, fields in VALIDITY.items():
            for field in fields:
                require(bool(np.all(_payload_zero(rows[rows[flag] == 0], field))), f"video_overlay_payload_not_zero:{field}")
        for field in ("position_arena_local_px", "position_camera_px", "heading_camera_angle_degrees",
                      "heading_origin_camera_px", "arrow_origin_arena_local_px", "arrow_direction_canvas"):
            require(bool(np.all(np.isfinite(rows[field]))), f"video_overlay_nonfinite:{field}")

        # invariant 0: the encoded frame index is the frame table's.
        for value, index, valid in zip(frame.tolist(), rows["video_frame_index"].tolist(), rows["video_frame_index_valid"].tolist()):
            require(value in frames, "video_overlay_coverage_gap:frame_unresolved")
            expected_index, expected_valid = frames[value]
            require(
                valid == expected_valid and (not valid or index == expected_index),
                "video_overlay_video_frame_index_mismatch",
            )

        decision, drawn = rows["decision_draw"] == 1, rows["drawn"] == 1
        used = rows["input_used"].astype(np.int64)
        require(bool(np.all(np.isin(used, (INPUT_NONE, INPUT_CURRENT, INPUT_HELD)))), "video_overlay_input_used_unknown")
        # invariant 1
        require(bool(np.all(drawn == (decision & (rows["video_frame_index_valid"] == 1)))), "video_overlay_drawn_flag")
        # invariant 2
        require(bool(np.all(~decision | ((rows["position_valid"] == 1) & (used >= 1)))), "video_overlay_decision_flag")
        # Citrus Decide: draw iff a box gave a valid position, and for a held
        # box only under held_policy draw.
        if settings["held_policy"] == "skip":
            require(bool(np.all(~(decision & (used == INPUT_HELD)))), "video_overlay_held_policy_skip")
            allowed = used == INPUT_CURRENT
        else:
            allowed = used >= INPUT_CURRENT
        require(bool(np.all(decision == ((rows["position_valid"] == 1) & allowed))), "video_overlay_decision_flag")
        # invariant 3
        none = used == INPUT_NONE
        for field in ("position_valid", "source_camera_id_valid", "box_sequence_id_valid", "box_recording_frame_id_valid"):
            require(bool(np.all(rows[field][none] == 0)), f"video_overlay_no_input_not_empty:{field}")
        for field in ("position_camera_px", "box_index_in_payload", "box_payload_timestamp_ns_monotonic"):
            require(bool(np.all(_payload_zero(rows[none], field))), f"video_overlay_payload_not_zero:{field}")
        # invariant 4
        some = ~none
        for field in ("source_camera_id_valid", "box_sequence_id_valid", "box_recording_frame_id_valid"):
            require(bool(np.all(rows[field][some] == 1)), f"video_overlay_box_identity_invalid:{field}")
        require(
            bool(np.all(rows["source_camera_id"][some] == np.uint64(shaman_camera))),
            "video_overlay_source_camera_mismatch",
        )
        # invariant 6
        for field, last in (("box_sequence_id", last_box), ("pose_sequence_id", last_pose)):
            values = rows[field][rows[field + "_valid"] == 1].astype(np.int64)
            require(
                values.size == 0 or (values[0] >= last and bool(np.all(np.diff(values) >= 0))),
                f"video_overlay_sequence_order:{field}",
            )
        box_seq = rows["box_sequence_id"][rows["box_sequence_id_valid"] == 1]
        pose_seq = rows["pose_sequence_id"][rows["pose_sequence_id_valid"] == 1]
        if box_seq.size:
            last_box = int(box_seq[-1])
        if pose_seq.size:
            last_pose = int(pose_seq[-1])
        # invariant 7
        heading = rows["heading_valid"] == 1
        midline_ok = rows["midline_valid"] == 1
        require(bool(np.all(heading == (rows["heading_status"] == HEADING_VALID))), "video_overlay_heading_flag")
        require(bool(np.all(~midline_ok | heading)), "video_overlay_midline_without_heading")
        # invariant 9
        require(bool(np.all(rows["heading_status"][used == INPUT_HELD] == 0)), "video_overlay_held_heading")
        # invariant 10
        require(
            bool(np.all((rows["arrow_drawn"] == 1) == (drawn & midline_ok & settings["arrow_active"]))),
            "video_overlay_arrow_flag",
        )
        # invariant 11
        if not pose_present:
            require(bool(np.all(rows["heading_status"] == 0)), "video_overlay_heading_without_pose")
        # heading_valid needs the slot the heading came from.
        require(bool(np.all(rows["pose_sequence_id_valid"][heading] == 1)), "video_overlay_heading_without_pose_slot")

        superseded += _check_boxes(h5, rows, homography, origin)
        _check_headings(h5, rows, homography, origin)
    return {"rows": total, "superseded_box_rows": superseded}


def _check_boxes(h5, rows, homography, origin) -> int:
    """Invariant 5 and the exact dot position; returns superseded rows."""

    some = np.flatnonzero(rows["input_used"] >= 1)
    keys = {
        (int(rows["source_camera_id"][i]), int(rows["box_recording_frame_id"][i]), int(rows["box_index_in_payload"][i]))
        for i in some
    }
    boxes = _box_rows(h5, keys)
    superseded = 0
    exact_rows, exact_boxes = [], []
    for i in some:
        key = (int(rows["source_camera_id"][i]), int(rows["box_recording_frame_id"][i]), int(rows["box_index_in_payload"][i]))
        candidates = boxes.get(key) or []
        require(candidates, f"video_overlay_box_row_missing:frame={int(rows['stimulus_frame_num'][i])}")
        stamp = int(rows["box_payload_timestamp_ns_monotonic"][i])
        same = [box for box in candidates if box[0] == stamp]
        if same:
            require(len(same) == 1, "video_overlay_box_row_duplicate")
            exact_rows.append(i)
            exact_boxes.append(same[0])
        elif max(box[0] for box in candidates) > stamp:
            superseded += 1
        else:
            require(False, f"video_overlay_box_row_earlier:frame={int(rows['stimulus_frame_num'][i])}")
    if not exact_rows:
        return superseded
    index = np.asarray(exact_rows, dtype=np.int64)
    x_min = np.asarray([b[1] for b in exact_boxes], np.float32)
    y_min = np.asarray([b[2] for b in exact_boxes], np.float32)
    width = np.asarray([b[3] for b in exact_boxes], np.float32)
    height = np.asarray([b[4] for b in exact_boxes], np.float32)
    centre = np.stack([x_min + width / F32(2.0), y_min + height / F32(2.0)], axis=1)
    require(
        bool(np.all(_bits(rows["position_camera_px"][index]) == _bits(centre))),
        "video_overlay_position_mismatch:camera_px",
    )
    ok, cx, cy = box_arena_local(homography, origin, x_min, y_min, width, height)
    require(bool(np.all((rows["position_valid"][index] == 1) == ok)), "video_overlay_position_mismatch:position_valid")
    expected = np.stack([cx, cy], axis=1)
    require(
        bool(np.all(_bits(rows["position_arena_local_px"][index][ok]) == _bits(expected[ok]))),
        "video_overlay_position_mismatch:arena_local_px",
    )
    return superseded


def _check_headings(h5, rows, homography, origin) -> None:
    """Invariant 8 and the exact arrow origin and direction."""

    heading = np.flatnonzero(rows["heading_valid"] == 1)
    if heading.size == 0:
        return
    require(POSE_GROUP in h5, "video_overlay_heading_without_pose")
    poses = _pose_objects(h5, {int(rows["pose_sequence_id"][i]) for i in heading})
    origin_x = np.zeros(heading.size, np.float32)
    origin_y = np.zeros(heading.size, np.float32)
    direction_x = np.zeros(heading.size, np.float32)
    direction_y = np.zeros(heading.size, np.float32)
    for k, i in enumerate(heading):
        sequence = int(rows["pose_sequence_id"][i])
        pose = poses.get(sequence)
        require(pose is not None, "video_overlay_pose_update_missing")
        require(pose["camera_id"] == int(rows["source_camera_id"][i]), "video_overlay_pose_camera_mismatch")
        if rows["pose_recording_frame_id_valid"][i] == 1:
            require(pose["recording_frame_id"] == int(rows["pose_recording_frame_id"][i]), "video_overlay_pose_recording_frame_mismatch")
        index = int(rows["pose_object_index_in_payload"][i])
        require(index == int(rows["box_index_in_payload"][i]), "video_overlay_pose_object_index_mismatch")
        found = pose["objects"].get(index)
        require(found is not None and int(found["heading_status"]) == HEADING_VALID, "video_overlay_pose_object_not_valid")
        require(
            _bits(found["heading_angle_degrees"])[()] == _bits(rows["heading_camera_angle_degrees"][i])[()]
            and _bits(found["heading_origin_x_px"])[()] == _bits(rows["heading_origin_camera_px"][i][0])[()]
            and _bits(found["heading_origin_y_px"])[()] == _bits(rows["heading_origin_camera_px"][i][1])[()],
            "video_overlay_pose_heading_mismatch",
        )
        origin_x[k], origin_y[k] = found["heading_origin_x_px"], found["heading_origin_y_px"]
        direction_x[k], direction_y[k] = found["heading_direction_x"], found["heading_direction_y"]
    ok, ax, ay, ux, uy = midline(homography, origin, origin_x, origin_y, direction_x, direction_y)
    stored = rows["midline_valid"][heading] == 1
    require(bool(np.all(stored == ok)), "video_overlay_arrow_mismatch:midline_valid")
    if not ok.any():
        return
    sel = heading[ok]
    require(
        bool(np.all(_bits(rows["arrow_origin_arena_local_px"][sel]) == _bits(np.stack([ax[ok], ay[ok]], axis=1)))),
        "video_overlay_arrow_mismatch:origin",
    )
    require(
        bool(np.all(_bits(rows["arrow_direction_canvas"][sel]) == _bits(np.stack([ux[ok], uy[ok]], axis=1)))),
        "video_overlay_arrow_mismatch:direction",
    )
    norm = np.hypot(rows["arrow_direction_canvas"][sel][:, 0].astype(np.float64), rows["arrow_direction_canvas"][sel][:, 1].astype(np.float64))
    require(bool(np.all(np.abs(norm - 1.0) <= UNIT_TOLERANCE)), "video_overlay_arrow_direction_not_unit")


__all__ = [
    "COMPONENT",
    "SETTINGS",
    "STATES",
    "box_arena_local",
    "midline",
    "project",
    "quantise_rgba8",
    "validate_settings",
    "validate_video_overlay_states",
]
