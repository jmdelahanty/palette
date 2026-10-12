"""Fish heading and left/right: the convention Palette and Citrus share.

agent-contracts ``citrus-palette/heading/heading_convention_v1.json`` (merged
6b63aa2), vendored with its reference cases in ``fisheye/shared/contracts``.
Source-camera pixels, top-left origin, y down. Forward runs from the swim
bladder to the eye midpoint; the heading is ``degrees(atan2(-forward_y,
forward_x))`` (0 = image right, +90 = image up). Its range is [-180, 180] and
both ends are the same heading, so compare headings only through
``wrap_delta_degrees``. Anatomical left is ``(forward_y, -forward_x)``.

New code computes headings, left axes and heading differences here instead of
re-deriving them.
"""

from __future__ import annotations

import numpy as np

ANGLE_CONVENTION = "atan2_negative_y_x_degrees"
HANDEDNESS = "forward_cross_left_negative_source_camera_xy"


def heading_degrees(forward_x, forward_y) -> np.ndarray:
    """Heading of a forward vector in y-down pixels; never normalises -180."""

    return np.degrees(np.arctan2(-np.asarray(forward_y, dtype=np.float64), np.asarray(forward_x, dtype=np.float64)))


def left_axis(forward_x, forward_y) -> tuple[np.ndarray, np.ndarray]:
    """Anatomical left for a y-down forward vector (determinant -1)."""

    return np.asarray(forward_y, dtype=np.float64), -np.asarray(forward_x, dtype=np.float64)


def wrap_delta_degrees(delta) -> np.ndarray:
    """Signed angular difference in [-180, 180); a difference of 360 is 0."""

    return (np.asarray(delta, dtype=np.float64) + 180.0) % 360.0 - 180.0


__all__ = ["ANGLE_CONVENTION", "HANDEDNESS", "heading_degrees", "left_axis", "wrap_delta_degrees"]
