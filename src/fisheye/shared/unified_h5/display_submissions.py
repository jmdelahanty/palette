"""Citrus display-submission timing per stimulus state, from an admitted unified H5.

Citrus logs one row per display frame in
``/timing/display_submissions/frame_submissions``: which stimulus state was
composited into that frame and when ``glfwSwapBuffers`` returned (CLOCK_REALTIME
``swap_call_end_ns_epoch`` and the session steady clock). With vsync off this is
the moment the state was handed to the display chain, not photon time; the
state is fully visible within about two projector refreshes plus projector
latency. Held frames repeat a state; some logged states are never submitted.

This module reduces those rows to one record per ``/frames/stimulus`` state:
the first submission (earliest the state could be on screen), how many display
frames carried it, and whether it was ever submitted. It reads only through the
sealed reference (``UnifiedSource``), so every block is digest-verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .common import require
from .metadata import string_attributes
from .reference import UnifiedSource

DISPLAY_SUBMISSIONS_PATH = "/timing/display_submissions/frame_submissions"
DISPLAY_TIMING_PATH = "/timing/display_submissions"
STIMULUS_FRAMES_PATH = "/frames/stimulus"

TABLE_SCHEMA = {
    "schema_id": "citrus.experimental_h5.display_submission",
    "schema_version": 1,
    "source_schema_id": "citrus.display_frame_submission",
    "source_schema_version": 1,
}
PARENT_SCHEMA = {"schema_id": "citrus.display_timing", "schema_version": 1}
NOT_SUBMITTED = np.iinfo(np.int64).min


@dataclass(frozen=True)
class StimulusDisplaySubmissions:
    """One row per ``/frames/stimulus`` state, in stimulus frame order."""

    stimulus_frame_num: np.ndarray
    submitted: np.ndarray
    first_swap_end_ns_epoch: np.ndarray
    first_swap_end_ns_session: np.ndarray
    first_display_frame_index: np.ndarray
    submission_count: np.ndarray
    invalid_prefix_rows: int
    display_rate_target_hz: float
    glfw_swap_interval: int
    timing_semantics: str
    source_reference_sha256: str

    @property
    def never_submitted_count(self) -> int:
        return int(np.count_nonzero(~self.submitted))

    @property
    def held_state_count(self) -> int:
        return int(np.count_nonzero(self.submission_count > 1))


def _scalar_number(descriptor: Mapping[str, Any], name: str) -> int | float:
    spec = descriptor.get("type", {})
    require(descriptor.get("shape") == [], f"attribute_not_scalar:{name}")
    raw = bytes.fromhex(descriptor.get("payload_hex", ""))
    if spec.get("class") == "integer":
        require(len(raw) == spec.get("size_bytes"), f"attribute_size_mismatch:{name}")
        return int.from_bytes(raw, "little", signed=bool(spec.get("signed")))
    require(
        spec.get("class") == "ieee_float" and spec.get("size_bytes") == 8,
        f"attribute_not_number:{name}",
    )
    return float(np.frombuffer(raw, dtype="<f8")[0])


def _decode(descriptors: Mapping[str, Any], *, strings: tuple[str, ...], numbers: tuple[str, ...]) -> dict:
    for name in (*strings, *numbers):
        require(name in descriptors, f"display_timing_attribute_missing:{name}")
    decoded = string_attributes({name: descriptors[name] for name in strings})
    decoded.update({name: _scalar_number(descriptors[name], name) for name in numbers})
    return decoded


def build_stimulus_display_submissions(
    rows: np.ndarray,
    stimulus_frame_num: np.ndarray,
    *,
    table_attrs: Mapping[str, Any],
    parent_attrs: Mapping[str, Any],
    source_reference_sha256: str,
) -> StimulusDisplaySubmissions:
    """Validate display-submission rows and reduce them to one record per state."""

    for name, expected in TABLE_SCHEMA.items():
        require(table_attrs.get(name) == expected, f"display_submission_schema:{name}")
    for name, expected in PARENT_SCHEMA.items():
        require(parent_attrs.get(name) == expected, f"display_timing_schema:{name}")

    states = np.asarray(stimulus_frame_num, dtype=np.int64)
    require(states.size > 0 and bool(np.all(np.diff(states) > 0)), "stimulus_frames_not_strictly_increasing")

    valid = np.asarray(rows["source_state_valid"]).astype(bool)
    first_valid = int(np.argmax(valid)) if valid.any() else valid.size
    # Citrus: rows are invalid only before the arena published its first state.
    require(bool(valid[first_valid:].all()), "display_submission_invalid_row_after_first_state")
    display_index = np.asarray(rows["display_frame_index"], dtype=np.int64)
    require(bool(np.all(np.diff(display_index) > 0)), "display_frame_index_not_strictly_increasing")

    kept = rows[first_valid:]
    stim = np.asarray(kept["source_stimulus_frame_num"], dtype=np.int64)
    epoch = np.asarray(kept["swap_call_end_ns_epoch"], dtype=np.int64)
    session = np.asarray(kept["swap_call_end_ns_session"], dtype=np.int64)
    require(bool(np.all(np.diff(stim) >= 0)), "display_submission_states_out_of_order")
    require(bool(np.all(np.diff(epoch) >= 0)), "display_submission_epoch_time_decreases")
    require(bool(np.all(np.diff(session) >= 0)), "display_submission_session_time_decreases")

    unique, first_row, counts = np.unique(stim, return_index=True, return_counts=True)
    position = np.searchsorted(states, unique)
    require(
        bool(np.all(position < states.size)) and bool(np.array_equal(states[np.minimum(position, states.size - 1)], unique)),
        "display_submission_state_not_in_frames",
    )

    n = states.size
    submitted = np.zeros(n, dtype=bool)
    first_epoch = np.full(n, NOT_SUBMITTED, dtype=np.int64)
    first_session = np.full(n, NOT_SUBMITTED, dtype=np.int64)
    first_display = np.full(n, -1, dtype=np.int64)
    submission_count = np.zeros(n, dtype=np.int32)
    submitted[position] = True
    first_epoch[position] = epoch[first_row]
    first_session[position] = session[first_row]
    first_display[position] = np.asarray(kept["display_frame_index"], dtype=np.int64)[first_row]
    submission_count[position] = counts

    return StimulusDisplaySubmissions(
        stimulus_frame_num=states,
        submitted=submitted,
        first_swap_end_ns_epoch=first_epoch,
        first_swap_end_ns_session=first_session,
        first_display_frame_index=first_display,
        submission_count=submission_count,
        invalid_prefix_rows=first_valid,
        display_rate_target_hz=float(parent_attrs["display_submission_rate_target_hz"]),
        glfw_swap_interval=int(parent_attrs["glfw_swap_interval"]),
        timing_semantics=str(table_attrs["timing_semantics"]),
        source_reference_sha256=str(source_reference_sha256),
    )


def load_stimulus_display_submissions(source: UnifiedSource) -> StimulusDisplaySubmissions:
    """Read and reduce the display-submission table of one admitted unified H5."""

    table_attrs = _decode(
        source.typed_attributes(DISPLAY_SUBMISSIONS_PATH),
        strings=("schema_id", "source_schema_id", "timing_semantics"),
        numbers=("schema_version", "source_schema_version"),
    )
    parent_attrs = _decode(
        source.typed_attributes(DISPLAY_TIMING_PATH),
        strings=("schema_id",),
        numbers=("schema_version", "display_submission_rate_target_hz", "glfw_swap_interval"),
    )
    frames = source.read_table(STIMULUS_FRAMES_PATH)
    return build_stimulus_display_submissions(
        source.read_table(DISPLAY_SUBMISSIONS_PATH),
        frames["stimulus_frame_num"],
        table_attrs=table_attrs,
        parent_attrs=parent_attrs,
        source_reference_sha256=source.reference_sha256,
    )
