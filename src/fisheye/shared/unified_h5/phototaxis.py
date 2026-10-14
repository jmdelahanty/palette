"""Reactive phototaxis states (admission revision 6, phototaxis_v1).

``/components/phototaxis/states`` is the per-update decision witness of a
heading-split step: which input drove the drawing (the raw pose heading, a
held heading, or a fallback). Its contract is agent-contracts
``citrus-palette/reactive_phototaxis/reactive_phototaxis_states_v1.json``
(vendored as ``contracts/reactive_phototaxis_states_v1.json``); the catalog
validates layout, this module validates the contract's invariants against the
trial index, the execution index and the pose observations. Recorded state is
the authority: nothing here re-runs the decision rule except the hysteresis
inequality the contract states.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fisheye.shared.heading_convention import wrap_delta_degrees

from .common import require
from .hdf5_types import iter_blocks
from .pose import BINDING
from .pose import CONTRACT as POSE_CONTRACT
from .pose import GROUP as POSE_GROUP
from .pose import _contract_version as pose_contract_version
from .pose import _json as pose_json

STATES = "/components/phototaxis/states"
HEADING_SPLIT = "heading_split"
HEADING_VALID = 7
INPUT_POSE, INPUT_HELD, INPUT_FALLBACK = 1, 2, 3
FALLBACK_IDS = {"static_fill": 1, "uniform_side_a": 2, "uniform_side_b": 3}
# Every field with explicit validity holds payload zero when its flag is 0.
VALIDITY = {
    "source_camera_id_valid": ("source_camera_id",),
    "pose_sequence_id_valid": ("pose_sequence_id",),
    "box_sequence_id_valid": ("box_sequence_id",),
    "pose_recording_frame_id_valid": ("pose_recording_frame_id",),
    "heading_valid": ("heading_camera_angle_degrees", "heading_origin_camera_px"),
    "midline_valid": (
        "midline_origin_canvas_px",
        "midline_direction_canvas",
        "midline_left_canvas",
        "midline_angle_degrees_canvas",
    ),
    "held_valid": (
        "held_camera_angle_degrees",
        "held_origin_canvas_px",
        "held_direction_canvas",
        "held_left_canvas",
        "held_angle_degrees_canvas",
    ),
    "pose_identity_valid": ("pose_model_hash64", "pose_skeleton_hash64"),
}
RAW_TO_HELD = {
    "midline_origin_canvas_px": "held_origin_canvas_px",
    "midline_direction_canvas": "held_direction_canvas",
    "midline_left_canvas": "held_left_canvas",
    "midline_angle_degrees_canvas": "held_angle_degrees_canvas",
    "heading_camera_angle_degrees": "held_camera_angle_degrees",
}


def reactive_steps(snapshot) -> dict[int, Mapping]:
    """Trial-index step index -> ``features.reactive`` of each heading-split step."""

    steps = {}
    for step in snapshot.trial_index_payload["steps"]:
        features = step["features"]
        fill = features.get("fill")
        if isinstance(fill, Mapping) and fill.get("type") == HEADING_SPLIT:
            steps[int(step["step_index"])] = features["reactive"]
    return steps


def _prefix64(digest: object) -> int:
    require(
        isinstance(digest, str) and len(digest) == 64 and digest == digest.lower(),
        "phototaxis_pose_digest_malformed",
    )
    return int(digest[:16], 16)


def _pose_identity(h5, camera_serial: str) -> tuple[int, int]:
    metadata = pose_json(h5, POSE_CONTRACT).get("metadata_cameras")
    camera = metadata.get(camera_serial) if isinstance(metadata, Mapping) else None
    require(isinstance(camera, Mapping), "phototaxis_pose_camera_metadata_missing")
    return _prefix64(camera.get("engine_sha256")), _prefix64(camera.get("sidecar_sha256"))


def _valid_pose_headings(h5, wanted: set[int]):
    """Status-7 object headings and the camera id of each wanted pose update."""

    updates, objects = h5[POSE_GROUP + "/updates"], h5[POSE_GROUP + "/objects"]
    rows: dict[int, int] = {}
    cameras: dict[int, int] = {}
    for start, block in iter_blocks(updates):
        hits = np.isin(block["sequence_id"], list(wanted))
        for offset in np.flatnonzero(hits):
            sequence = int(block["sequence_id"][offset])
            require(sequence not in cameras, "phototaxis_pose_sequence_duplicate")
            rows[start + int(offset)] = sequence
            cameras[sequence] = int(block["camera_id"][offset])
    headings: dict[int, list[tuple[float, float, float]]] = {}
    for start, block in iter_blocks(objects):
        valid = np.isin(block["update_row"], list(rows)) & (
            block["heading_status"] == HEADING_VALID
        )
        for offset in np.flatnonzero(valid):
            row = block[offset]
            headings.setdefault(rows[int(row["update_row"])], []).append(
                (
                    float(row["heading_angle_degrees"]),
                    float(row["heading_origin_x_px"]),
                    float(row["heading_origin_y_px"]),
                )
            )
    return headings, cameras


def _in_named_step(execution, frames: np.ndarray, step_index: np.ndarray) -> np.ndarray:
    """Whether each row's frame lies in the executed interval of its own step.

    ``protocol_step_index`` is authoritative. Citrus logs protocol events with
    the previous update's frame, so consecutive half-open intervals overlap by
    exactly one frame and a step's first row is at start + 1: membership is
    tested against the named step's interval, never by a unique containing one.
    """

    intervals = {}
    for step in execution.steps:
        require(step.step_index not in intervals, "phototaxis_execution_step_duplicate")
        intervals[step.step_index] = step.interval
    inside = np.zeros(frames.shape, dtype=bool)
    for index in np.unique(step_index):
        interval = intervals.get(int(index))
        if interval is None:
            continue
        here = step_index == index
        inside[here] = (frames[here] >= interval.start_stimulus_frame_inclusive) & (
            frames[here] < interval.end_stimulus_frame_exclusive
        )
    return inside


def _rows_equal(rows, left: str, right: str) -> np.ndarray:
    a, b = rows[left], rows[right]
    equal = a == b
    return equal.reshape(len(rows), -1).all(axis=1)


def _payload_zero(rows, name: str) -> np.ndarray:
    return (rows[name] == 0).reshape(len(rows), -1).all(axis=1)


def validate_phototaxis_states(h5, snapshot, execution, *, camera_serial: str) -> int:
    """Validate the states table against its contract; return its row count.

    A file without a heading-split step carries no rows. A file with one
    carries the table and pose observation v2 (its completeness is the pose
    validator's); every row then satisfies the states contract invariants.
    """

    steps = reactive_steps(snapshot)
    if not steps:
        require(STATES not in h5 or h5[STATES].shape[0] == 0, "phototaxis_states_without_reactive_step")
        return 0
    require(STATES in h5, "reactive_phototaxis_states_missing")
    require(
        POSE_GROUP in h5 and pose_contract_version(h5) == 2,
        "reactive_phototaxis_requires_pose_v2",
    )
    model64, skeleton64 = _pose_identity(h5, camera_serial)
    # The pose table gates headings with its contract's confidence; a step gate
    # below it could drive the split from a heading the table marks status 5.
    pose_gate = pose_json(h5, POSE_CONTRACT).get("minimum_keypoint_confidence")
    require(isinstance(pose_gate, (int, float)) and not isinstance(pose_gate, bool), "phototaxis_pose_gate_missing")
    for reactive in steps.values():
        require(
            float(reactive["minimum_keypoint_confidence"]) >= float(pose_gate),
            "phototaxis_step_gate_below_pose_gate",
        )
    shaman_camera = pose_json(h5, BINDING).get("shaman_numeric_camera_id")

    total, last_frame = 0, -1
    source_camera: int | None = None
    pose_rows: dict[int, tuple[float, float, float]] = {}
    for _start, rows in iter_blocks(h5[STATES]):
        frames = rows["stimulus_frame_num"].astype(np.int64)
        require(
            frames.size == 0 or (frames[0] > last_frame and np.all(np.diff(frames) > 0)),
            "phototaxis_states_frame_order",
        )
        if frames.size:
            last_frame = int(frames[-1])
        total += len(rows)

        for flag, fields in VALIDITY.items():
            require(np.isin(rows[flag], (0, 1)).all(), f"phototaxis_validity_not_boolean:{flag}")
            invalid = rows[flag] == 0
            for name in fields:
                require(
                    _payload_zero(rows, name)[invalid].all(),
                    f"phototaxis_invalid_payload_nonzero:{name}",
                )

        status = rows["heading_status"]
        heading = rows["heading_valid"] == 1
        midline = rows["midline_valid"] == 1
        held = rows["held_valid"] == 1
        used = rows["input_used"]
        fallback = rows["fallback_id"]
        age = rows["held_age_updates"]
        require(np.all(status <= 9), "phototaxis_heading_status_unknown")
        require(np.array_equal(heading, status == HEADING_VALID), "phototaxis_heading_valid_status")
        require(np.all(heading[midline]), "phototaxis_midline_without_heading")
        require(np.isin(used, (INPUT_POSE, INPUT_HELD, INPUT_FALLBACK)).all(), "phototaxis_input_used_unknown")
        require(np.isin(fallback, (0, 1, 2, 3)).all(), "phototaxis_fallback_unknown")
        require(np.array_equal(held, np.isin(used, (INPUT_POSE, INPUT_HELD))), "phototaxis_held_valid_input")
        require(np.array_equal(used == INPUT_FALLBACK, fallback != 0), "phototaxis_fallback_input")
        usable = heading & midline
        # A usable raw heading always yields a held midline with age 0.
        require(np.all(age[usable] == 0), "phototaxis_age_on_usable_heading")
        require(not np.any(usable & (used == INPUT_FALLBACK)), "phototaxis_fallback_with_usable_heading")

        pose = used == INPUT_POSE
        require(np.all(usable[pose]), "phototaxis_pose_input_without_heading")
        for raw, held_name in RAW_TO_HELD.items():
            require(_rows_equal(rows, raw, held_name)[pose].all(), f"phototaxis_pose_held_differs:{held_name}")
        require(np.all(rows["pose_sequence_id_valid"][pose] == 1), "phototaxis_pose_input_without_slot")

        hold = used == INPUT_HELD
        hysteresis = hold & usable
        require(
            _rows_equal(rows, "midline_origin_canvas_px", "held_origin_canvas_px")[hysteresis].all(),
            "phototaxis_hysteresis_origin",
        )
        require(np.all(age[hold & ~usable] >= 1), "phototaxis_lost_pose_hold_age")

        step_index = rows["protocol_step_index"].astype(np.int64)
        require(_in_named_step(execution, frames, step_index).all(), "phototaxis_step_not_active")
        for index in np.unique(step_index):
            require(int(index) in steps, "phototaxis_step_not_heading_split")
            reactive = steps[int(index)]
            here = step_index == index
            delta = np.abs(
                wrap_delta_degrees(
                    rows["heading_camera_angle_degrees"].astype(np.float64)
                    - rows["held_camera_angle_degrees"].astype(np.float64)
                )
            )
            require(
                np.all(delta[here & hysteresis] < float(reactive["minimum_heading_displacement_degrees"])),
                "phototaxis_hysteresis_threshold",
            )
            limit = reactive["maximum_held_heading_updates"]
            if limit != "never":
                require(np.all(age[here & hold] <= int(limit)), "phototaxis_hold_beyond_limit")
            require(
                np.all(fallback[here & (used == INPUT_FALLBACK)] == FALLBACK_IDS[reactive["unknown_heading_fallback"]]),
                "phototaxis_fallback_differs_from_step",
            )

        identity = rows["pose_identity_valid"] == 1
        require(
            np.all(rows["pose_model_hash64"][identity] == np.uint64(model64))
            and np.all(rows["pose_skeleton_hash64"][identity] == np.uint64(skeleton64)),
            "phototaxis_pose_identity_mismatch",
        )
        cameras = np.unique(rows["source_camera_id"][rows["source_camera_id_valid"] == 1])
        require(cameras.size <= 1, "phototaxis_source_camera_not_unique")
        if cameras.size:
            require(source_camera in (None, int(cameras[0])), "phototaxis_source_camera_not_unique")
            source_camera = int(cameras[0])
            require(
                type(shaman_camera) is int and source_camera == shaman_camera,
                "phototaxis_source_camera_not_binding",
            )

        for offset in np.flatnonzero(pose):
            row = rows[offset]
            pose_rows[int(row["pose_sequence_id"])] = (
                float(row["heading_camera_angle_degrees"]),
                float(row["heading_origin_camera_px"][0]),
                float(row["heading_origin_camera_px"][1]),
                int(row["source_camera_id"]) if row["source_camera_id_valid"] else None,
            )

    headings, cameras = _valid_pose_headings(h5, set(pose_rows))
    for sequence, (*expected, camera) in pose_rows.items():
        require(tuple(expected) in headings.get(sequence, ()), "phototaxis_pose_row_missing")
        require(camera is None or cameras.get(sequence) == camera, "phototaxis_source_camera_not_pose_slot")
    return total


__all__ = ["STATES", "reactive_steps", "validate_phototaxis_states"]
