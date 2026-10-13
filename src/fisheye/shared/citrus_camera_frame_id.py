"""Citrus camera frame ids are Orange 1-based ``recording_frame_id`` values.

Citrus stamps stimulus rows and events with the last camera frame id Orange
published over IPC (``triggering_camera_frame_id``, events ``camera_frame_id``).
Measured on 2026-10-11 across every legacy recording with bounding-box payloads,
that id equals the 1-based ``recording_frame_id``; 0 means no frame was known.
Palette frame axes are 0-based ``parent_frame_index``, so readers convert with
``id - 1`` here instead of indexing by the raw id (which lands one frame late).

Legacy H5s only. In unified (v2) H5s the trigger columns are Citrus's IPC frame
counter, which is offset from ``recording_frame_id`` per session; use
``/correspondence/frames/sources`` there instead.
"""

from __future__ import annotations

from typing import Any

CITRUS_CAMERA_FRAME_ID_RULE = "citrus_camera_frame_id_is_recording_frame_id_minus_1.v1"


def parent_frame_index_from_citrus_camera_frame_id(value: Any) -> int | None:
    """Return the 0-based parent frame index, or ``None`` when the id is unknown.

    Citrus writes 0 before it has received any camera frame id; negative values
    are sentinels. Neither names a frame.
    """

    frame_id = int(value)
    return frame_id - 1 if frame_id >= 1 else None
