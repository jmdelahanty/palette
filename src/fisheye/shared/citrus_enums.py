"""
Citrus enum definitions for event types and stimulus modes.

The **source of truth** for these enums is the H5/zarr data itself — each
recording carries its own enum tables under ``analysis/enums/`` (written by
``import_stimulus_to_zarr``).  Use :func:`load_event_types`,
:func:`load_stimulus_modes`, or :func:`load_chaser_loom_modes` to read from
the data at runtime.

The module-level dicts (``EXPERIMENT_EVENT_TYPE``, ``STIMULUS_MODE``, etc.)
are **hardcoded fallbacks** for use when the data doesn't carry its own
enum tables (old recordings, synthetic test data, offline scripts).  They
will drift if Citrus adds new event types — the loaders will not.

**Prefer string-based matching** (``event_name == "STEP_START"``) over
int-based (``event_type == 11``) since string names are stable across
encoding schemes.

See ``docs/experiment_types_reference.md`` for the full reference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Event types (modern Citrus encoding)
# ---------------------------------------------------------------------------

EXPERIMENT_EVENT_TYPE: Dict[int, str] = {
    0: "PROTOCOL_START",
    1: "PROTOCOL_STOP",
    2: "PROTOCOL_PAUSE",
    3: "PROTOCOL_RESUME",
    4: "PROTOCOL_FINISH",
    5: "PROTOCOL_CLEAR",
    6: "PROTOCOL_LOAD",
    7: "STEP_ADD",
    8: "STEP_REMOVE",
    9: "STEP_MOVE_UP",
    10: "STEP_MOVE_DOWN",
    11: "STEP_START",
    12: "STEP_END",
    13: "ITI_START",
    14: "ITI_END",
    15: "PARAMS_APPLIED",
    16: "MANAGER_REINIT",
    17: "MANAGER_REINIT_FAIL",
    18: "LOOM_AUTO_REPEAT_TRIGGER",
    19: "LOOM_MANUAL_START",
    20: "USER_INTERVENTION",
    21: "ERROR_RUNTIME",
    22: "LOG_MESSAGE",
    23: "IPC_BOUNDING_BOX_RECEIVED",
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END",
    29: "CHASER_RANDOM_TARGET_SET",
}

EVENT_NAME_TO_ID: Dict[str, int] = {
    name: idx for idx, name in EXPERIMENT_EVENT_TYPE.items()
}

# Convenience constants for the most commonly matched events.
EVENT_STEP_START = "STEP_START"
EVENT_STEP_END = "STEP_END"
EVENT_STEP_START_ID = EVENT_NAME_TO_ID[EVENT_STEP_START]
EVENT_STEP_END_ID = EVENT_NAME_TO_ID[EVENT_STEP_END]

EVENT_LOOM_AUTO_REPEAT = "LOOM_AUTO_REPEAT_TRIGGER"
EVENT_LOOM_MANUAL_START = "LOOM_MANUAL_START"
EVENT_LOOM_AUTO_REPEAT_ID = EVENT_NAME_TO_ID[EVENT_LOOM_AUTO_REPEAT]
EVENT_LOOM_MANUAL_START_ID = EVENT_NAME_TO_ID[EVENT_LOOM_MANUAL_START]

# ---------------------------------------------------------------------------
# Stimulus modes (from Citrus C++ StimulusMode::Type)
# ---------------------------------------------------------------------------

STIMULUS_MODE: Dict[int, str] = {
    -1: "UNDEFINED",
    2: "COHERENT_DOTS",
    3: "MOVING_GRATING",
    4: "SOLID_BLACK",
    5: "SOLID_WHITE",
    6: "CONCENTRIC_GRATING",
    7: "LOOMING_DOT",
    8: "STATIC_IMAGE",
    9: "CALIBRATION_GRID",
    10: "ARENA_DEFINITION_SQUARE",
    11: "SPOTLIGHT",
    12: "CHASER",
    13: "CALIBRATION_TEST_SHAPE",
    14: "SCROLLING_GRID",
    15: "INDEPENDENT_MOTION_GRID",
    16: "MOVING_DOTS",
    99: "NONE",
}

STIMULUS_MODE_NAME_TO_ID: Dict[str, int] = {
    name: idx for idx, name in STIMULUS_MODE.items()
}

# ---------------------------------------------------------------------------
# Chaser loom modes
# ---------------------------------------------------------------------------

CHASER_LOOM_MODE: Dict[int, str] = {
    0: "FIXED",
    1: "PROXIMITY",
    2: "VA_LOOM",
    3: "STATIONARY",
    4: "CAVE_DEF",
    5: "CAVE_AGG",
}

# ---------------------------------------------------------------------------
# Runtime loaders — read enums from zarr/H5 data (the source of truth)
# ---------------------------------------------------------------------------


def _decode_enum_name(value: Any) -> str:
    """Decode enum names from UTF-8 strings, bytes, or fixed-width uint8 rows."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, str):
        return value.rstrip("\x00")

    arr = np.asarray(value)
    if arr.dtype.kind in ("u", "i") and arr.ndim >= 1:
        payload = bytes(int(item) for item in arr.ravel() if int(item) != 0)
        return payload.decode("utf-8", errors="replace").rstrip("\x00")
    if arr.dtype.kind == "S":
        return bytes(arr.tobytes()).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value).rstrip("\x00")


def _load_enum_from_zarr(
    root: Any,
    path: str,
    fallback: Dict[int, str],
) -> Dict[int, str]:
    """Read an enum table from ``analysis/enums/<name>/`` in a zarr group.

    Returns the fallback dict if the enum group is missing or unreadable.
    """
    try:
        enum_group = root[path]
        ids = enum_group["id"][:]
        names = enum_group["name"][:]
        decoded_names = [_decode_enum_name(n) for n in names]
        return {int(i): n for i, n in zip(ids, decoded_names)}
    except (KeyError, TypeError, IndexError, ValueError):
        return dict(fallback)


def load_event_types(root: Optional[Any] = None) -> Dict[int, str]:
    """Load event type enum from zarr data, falling back to hardcoded dict."""
    if root is None:
        return dict(EXPERIMENT_EVENT_TYPE)
    # Modern imports preserve Citrus' enum table name: analysis/enums/events.
    # Older Palette docs/code sometimes used analysis/enums/event_types.
    for path in ("analysis/enums/events", "analysis/enums/event_types"):
        mapping = _load_enum_from_zarr(root, path, {})
        if mapping:
            return mapping
    return dict(EXPERIMENT_EVENT_TYPE)


def load_stimulus_modes(root: Optional[Any] = None) -> Dict[int, str]:
    """Load stimulus mode enum from zarr data, falling back to hardcoded dict."""
    if root is None:
        return dict(STIMULUS_MODE)
    return _load_enum_from_zarr(root, "analysis/enums/stimulus_modes", STIMULUS_MODE)


def load_chaser_loom_modes(root: Optional[Any] = None) -> Dict[int, str]:
    """Load chaser loom mode enum from zarr data, falling back to hardcoded dict."""
    if root is None:
        return dict(CHASER_LOOM_MODE)
    return _load_enum_from_zarr(root, "analysis/enums/chaser_loom_modes", CHASER_LOOM_MODE)


__all__ = [
    "EXPERIMENT_EVENT_TYPE",
    "EVENT_NAME_TO_ID",
    "EVENT_STEP_START",
    "EVENT_STEP_END",
    "EVENT_STEP_START_ID",
    "EVENT_STEP_END_ID",
    "EVENT_LOOM_AUTO_REPEAT",
    "EVENT_LOOM_MANUAL_START",
    "EVENT_LOOM_AUTO_REPEAT_ID",
    "EVENT_LOOM_MANUAL_START_ID",
    "STIMULUS_MODE",
    "STIMULUS_MODE_NAME_TO_ID",
    "CHASER_LOOM_MODE",
    "load_event_types",
    "load_stimulus_modes",
    "load_chaser_loom_modes",
]
