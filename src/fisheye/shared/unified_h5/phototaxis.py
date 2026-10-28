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

A raw heading drives an update only when it is *fresh*: the update consumed a
new SHAMAN slot (Citrus ``data_is_fresh``, true iff the update popped the
tracking queue; ``PhototaxisStimulus.cu`` ``raw.usable = tracking_fresh_ &&
HasHeading()``). Re-reading a slot is a held update whose age grows by one.
Jeremy adopted that meaning on 2026-10-19 (the contract text said any status-7
heading). Freshness is replayed as a change of ``pose_sequence_id``, the
identity of the slot the update read: inside a step from the previous row, at
a step's first row from the previous stimulus update's video tracking overlay
row (one per update in every mode). Where that row names no slot, or the file
has no overlay states, the step-first row is counted as unverified and its
freshness is taken from its own age, never assumed.
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
OVERLAY_STATES = "/components/video_tracking_overlay/states"
SESSION = "/metadata/session"
# Citrus builds whose writer set pose_identity_valid on every slot, including
# slots without a pose (both hashes 0, heading_status 0): Citrus main eab6b3b
# through 25c31f5 and the calibration builds 431dbde and 2719d0b (Citrus
# PhototaxisStimulus.cu FillFrameLogData). Jeremy 2026-10-19: for these builds
# only, such a row is read as identity-invalid, and the reading is recorded.
# A build from the Citrus fix on gets no exception; a pre-fix build used for
# recording that is not listed here is refused until it is added.
POSE_IDENTITY_NO_POSE_DEFECT = "citrus_pose_identity_valid_on_no_pose_slot_v1"
POSE_IDENTITY_DEFECT_BUILDS = frozenset(
    {
        "eab6b3bca118689cc7b25d984001c145eba1f816",
        "dd3986dacd5a0a61517163054436684d175a5d26",
        "1cf5fe6167f77739e4e01f9bcfdeee2359364688",
        "09172aa0f1b9e23fcbd3bd74f66bb25cb1215712",
        "f5213678a4d75118c93ef8187b1eab96c6e6f173",
        "4533043d2a596f46666f8633e74f2ffc960a221f",
        "e76917b18c29c6d09c2c124a3a8aa92dd5c4f94d",
        "3853f95b9375d08338da1def242751564ab21d4c",
        "920bf4164fd3399661fe99e85987fef0be686d57",
        "0693b808475f431589165858b990eeef475f9eb8",
        "5aef4b9628dc2e33aa54dd08f7ba9399d39914dd",
        "88132a3df50af446a89bd91b07ed1555021b658b",
        "90ccfd196b32fee9311d6c9e4c98e1b88633d3f2",
        "ede33c8f540ad915b89e1efaf1c5ae949925d196",
        "dcc6fb18b8a360a8501563d09e706a6820bf2c14",
        "2228a0b1c1fac0d42df1e1d68e7bb0bb06ac3a93",
        "dddbc916567695749143ca96767ed6d682f080f7",
        "c8c1162e5f0be317b6a986ff2000e270078acf9f",
        "49475be0ed5f614f89f18a4d14f5f9c78491f118",
        "25c31f545397ce388b4f996926d83852435adba3",
        "431dbde658b89635c028e43c459e8b0f464912ab",
        "2719d0ba6979cba0db5ef6d5aabd633150868339",
    }
)
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


def _producer_build(h5) -> tuple[str | None, bool | None]:
    if SESSION not in h5:
        return None, None
    attrs = h5[SESSION].attrs
    commit = attrs.get("git_commit_hash")
    if isinstance(commit, bytes):
        commit = commit.decode("utf-8")
    dirty = attrs.get("build_dirty")
    return (commit if isinstance(commit, str) else None), (
        None if dirty is None else bool(int(dirty))
    )


def _update_slots(h5):
    """Stimulus frame -> the slot that update read, for every update.

    From the video tracking overlay states (one row per stimulus update in every
    mode): ``pose_sequence_id``, or -1 where the row names none. None when the
    file carries no overlay states.
    """

    if OVERLAY_STATES not in h5:
        return None
    table = h5[OVERLAY_STATES]
    frames = table.fields("stimulus_frame_num")[:].astype(np.int64)
    valid = table.fields("pose_sequence_id_valid")[:] == 1
    slots = np.where(valid, table.fields("pose_sequence_id")[:].astype(np.int64), -1)
    require(frames.size == 0 or bool(np.all(np.diff(frames) > 0)), "phototaxis_overlay_frame_order")
    return frames, slots


def _previous_update_slot(update_slots, frames: np.ndarray) -> np.ndarray:
    """The slot each update's preceding stimulus update read (-1: not shown)."""

    all_frames, slots = update_slots
    at = np.searchsorted(all_frames, frames)
    require(
        bool(np.all(at < all_frames.size))
        and bool(np.all(all_frames[np.minimum(at, all_frames.size - 1)] == frames)),
        "phototaxis_update_without_overlay_row",
    )
    return np.where(at > 0, slots[np.maximum(at - 1, 0)], -1)


def _rows_equal(rows, left: str, right: str) -> np.ndarray:
    a, b = rows[left], rows[right]
    equal = a == b
    return equal.reshape(len(rows), -1).all(axis=1)


def _payload_zero(rows, name: str) -> np.ndarray:
    return (rows[name] == 0).reshape(len(rows), -1).all(axis=1)


def validate_phototaxis_states(
    h5, snapshot, execution, *, camera_serial: str, report: dict | None = None
) -> int:
    """Validate the states table against its contract; return its row count.

    A file without a heading-split step carries no rows. A file with one
    carries the table and pose observation v2 or v3 (its completeness is the pose
    validator's); every row then satisfies the states contract invariants.
    ``report``, when given, receives the step-first rows whose freshness the
    file cannot show and any producer defect read under a recorded rule.
    """

    steps = reactive_steps(snapshot)
    if not steps:
        require(STATES not in h5 or h5[STATES].shape[0] == 0, "phototaxis_states_without_reactive_step")
        return 0
    require(STATES in h5, "reactive_phototaxis_states_missing")
    require(
        POSE_GROUP in h5 and pose_contract_version(h5) in (2, 3),
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

    update_slots = _update_slots(h5)
    commit, dirty = _producer_build(h5)
    defect_build = dirty is False and commit in POSE_IDENTITY_DEFECT_BUILDS
    defect_rows = unverified = 0
    previous_row = None
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
        step_index = rows["protocol_step_index"].astype(np.int64)
        # The row before each row in this table (carried across blocks), and
        # whether it is the same step's previous stimulus update.
        if previous_row is None:
            prior_frame, prior_step = np.r_[-2, frames[:-1]], np.r_[-1, step_index[:-1]]
        else:
            prior_frame = np.r_[int(previous_row["stimulus_frame_num"]), frames[:-1]]
            prior_step = np.r_[int(previous_row["protocol_step_index"]), step_index[:-1]]
        continues = (prior_step == step_index) & (prior_frame == frames - 1)
        slot = np.where(rows["pose_sequence_id_valid"] == 1, rows["pose_sequence_id"].astype(np.int64), -1)
        prior_slot = np.r_[
            -1 if previous_row is None or not previous_row["pose_sequence_id_valid"]
            else int(previous_row["pose_sequence_id"]),
            slot[:-1],
        ]
        # A step's first row: the previous update wrote no states row.
        if update_slots is not None:
            prior_slot = np.where(continues, prior_slot, _previous_update_slot(update_slots, frames))
        known = continues | (prior_slot >= 0) if update_slots is not None else continues
        fresh = np.where(known, (slot >= 0) & (slot != prior_slot), age == 0)
        unverified += int(np.count_nonzero(~known))
        drives = usable & fresh
        # Age counts updates since the last fresh usable heading: 0 on one, else
        # one more than the same step's previous update (1 on a step's first
        # update, whose held heading starts empty).
        require(np.all(age[drives] == 0), "phototaxis_age_on_fresh_heading")
        prior_age = np.r_[
            -1 if previous_row is None else int(previous_row["held_age_updates"]), age[:-1].astype(np.int64)
        ]
        prior_used = np.r_[
            0 if previous_row is None else int(previous_row["input_used"]), used[:-1]
        ]
        stale = ~drives
        counted = stale & continues & (prior_used != INPUT_FALLBACK)
        require(np.all(age[counted].astype(np.int64) == prior_age[counted] + 1), "phototaxis_age_not_counted")
        require(np.all(age[stale & ~continues] == 1), "phototaxis_age_at_step_start")
        require(np.all(age[stale] >= 1), "phototaxis_age_without_fresh_heading")
        require(not np.any(drives & (used == INPUT_FALLBACK)), "phototaxis_fallback_with_fresh_heading")

        pose = used == INPUT_POSE
        require(np.all(drives[pose]), "phototaxis_pose_input_without_fresh_heading")
        for raw, held_name in RAW_TO_HELD.items():
            require(_rows_equal(rows, raw, held_name)[pose].all(), f"phototaxis_pose_held_differs:{held_name}")
        require(np.all(rows["pose_sequence_id_valid"][pose] == 1), "phototaxis_pose_input_without_slot")

        hold = used == INPUT_HELD
        hysteresis = hold & drives
        require(
            _rows_equal(rows, "midline_origin_canvas_px", "held_origin_canvas_px")[hysteresis].all(),
            "phototaxis_hysteresis_origin",
        )
        require(np.all(age[hold & ~usable] >= 1), "phototaxis_lost_pose_hold_age")

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
        if defect_build:
            no_pose = (
                identity
                & (rows["pose_model_hash64"] == 0)
                & (rows["pose_skeleton_hash64"] == 0)
                & (status == 0)
            )
            defect_rows += int(np.count_nonzero(no_pose))
            identity = identity & ~no_pose
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

        previous_row = rows[-1] if len(rows) else previous_row

    if report is not None:
        report["unverified_step_start_rows"] = unverified
        if defect_rows:
            report["producer_defects"] = {
                POSE_IDENTITY_NO_POSE_DEFECT: {"rows": defect_rows, "citrus_build": commit}
            }
    headings, cameras = _valid_pose_headings(h5, set(pose_rows))
    for sequence, (*expected, camera) in pose_rows.items():
        require(tuple(expected) in headings.get(sequence, ()), "phototaxis_pose_row_missing")
        require(camera is None or cameras.get(sequence) == camera, "phototaxis_source_camera_not_pose_slot")
    return total


__all__ = [
    "POSE_IDENTITY_DEFECT_BUILDS",
    "POSE_IDENTITY_NO_POSE_DEFECT",
    "STATES",
    "reactive_steps",
    "validate_phototaxis_states",
]
