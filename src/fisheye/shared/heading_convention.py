"""Fish heading and left/right: the convention Palette and Citrus share.

agent-contracts ``citrus-palette/heading/heading_convention_v1.json`` (merged
6b63aa2), vendored with its reference cases in ``fisheye/shared/contracts``.
Source-camera pixels, top-left origin, y down. Forward runs from the swim
bladder to the eye midpoint; the heading is ``degrees(atan2(-forward_y,
forward_x))`` (0 = image right, +90 = image up). Its range is [-180, 180] and
both ends are the same heading, so compare headings only through
``wrap_delta_degrees``. Anatomical left is ``(forward_y, -forward_x)``.

New code computes headings, left axes and heading differences here instead of
re-deriving them. ``scripts/check_heading_convention_access.py`` (CI) fails
on a new private copy of the heading formula or the angle wrap.

Every function takes a ``dtype`` (default float64) that is the precision the
arithmetic runs in. Producers that historically computed in float32 pass
``dtype=np.float32`` so their published values stay bit-identical.

Conventions that are not Palette's are named here too, with their exact
definitions, so that no module re-derives them privately:

``MEGABOUTS_HEAD_YAW`` (external tool)
    The head yaw the Megabouts-compatible tail-posture view publishes as
    ``head_yaw_rad`` (``fisheye.analysis.tail_posture_view_runs``):
    ``atan2(-v_y, -v_x)`` in radians for ``v = tail_base - head`` in
    source-camera pixels, i.e. ``atan2`` of the tail-base -> head vector with
    **no y flip** (positive = clockwise on screen, because y points down).
    It is a compatibility re-implementation of the public Megabouts keypoint
    contract (``megabouts.tracking_data.convert_tracking.
    compute_angles_from_keypoints``); Megabouts itself is not a dependency and
    nothing here asserts what Megabouts computes internally.
    ``fisheye.analysis.megabouts_convention_audit`` is the empirical check: it
    measures the *tail-angle* sign against a real Megabouts install (direct vs
    sign-flipped residuals) and only counts finite head-yaw values; it has not
    measured the head-yaw relation.

    Relation to the Palette heading: for one and the same vector ``u``,
    ``atan2(u_y, u_x) == -atan2(-u_y, u_x)`` exactly, so the Megabouts yaw of
    ``u`` is the negated Palette heading of ``u`` (in radians). But the two
    are defined on **different axes** (head -> tail base reversed vs swim
    bladder -> eye midpoint), which are not the same line in general, so there
    is no exact mapping from a published Palette heading to a Megabouts head
    yaw or back, and this module deliberately provides none.

``CHASER_CENTRED_BEARING`` (Palette-local)
    The bearing ``fisheye.analysis.chaser_escape_freeze_summary`` reports in
    the chaser-centred frame: millimetres, x right / **y up**, rotated so the
    chaser's motion heading points along +y. The bearing of ``(x, y)`` is
    ``degrees(atan2(x, y))``: 0 = straight ahead of the chaser, +90 = the
    chaser's right, i.e. **positive = clockwise** from the chaser heading,
    range [-180, 180]. The chaser heading that defines the frame is
    ``atan2(-dy_px, dx_px)`` of the chaser's frame-to-frame displacement in
    radians (Palette's angle convention, but of a motion vector, not a fish
    body axis).

Known non-conforming site, deliberately not migrated:
``fisheye.visualization.chaser_ring_traversal._chase_frame_heading`` draws the
fish heading in the chaser-centred frame and, where no body heading exists,
falls back to the direction of travel; the result is a drawing angle, not a
Palette heading.
"""

from __future__ import annotations

import numpy as np

ANGLE_CONVENTION = "atan2_negative_y_x_degrees"
HANDEDNESS = "forward_cross_left_negative_source_camera_xy"

MEGABOUTS_HEAD_YAW = "megabouts_head_yaw_atan2_tail_base_to_head_no_y_flip_radians"
CHASER_CENTRED_BEARING = "chaser_centred_clockwise_from_chaser_heading_y_up_mm_degrees"


def heading_degrees(forward_x, forward_y, *, dtype=np.float64) -> np.ndarray:
    """Heading of a forward vector in y-down pixels; never normalises -180."""

    return np.degrees(np.arctan2(-np.asarray(forward_y, dtype=dtype), np.asarray(forward_x, dtype=dtype)))


# The same formula for a vector that is not a fish forward axis (for example
# fish -> object), so call sites say what they measure.
image_angle_degrees = heading_degrees


def left_axis(forward_x, forward_y, *, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Anatomical left for a y-down forward vector (determinant -1)."""

    return np.asarray(forward_y, dtype=dtype), -np.asarray(forward_x, dtype=dtype)


def wrap_delta_degrees(delta, *, dtype=np.float64) -> np.ndarray:
    """Signed angular difference in [-180, 180); a difference of 360 is 0."""

    return (np.asarray(delta, dtype=dtype) + 180.0) % 360.0 - 180.0


def wrap_delta_radians(delta, *, dtype=np.float64) -> np.ndarray:
    """Signed angular difference in [-pi, pi); a difference of 2*pi is 0."""

    return (np.asarray(delta, dtype=dtype) + np.pi) % (2.0 * np.pi) - np.pi


def megabouts_head_yaw_radians(head_to_tail_base_x, head_to_tail_base_y) -> np.ndarray:
    """``MEGABOUTS_HEAD_YAW`` of ``v = tail_base - head`` (float64 radians)."""

    v_x = np.asarray(head_to_tail_base_x, dtype=np.float64)
    v_y = np.asarray(head_to_tail_base_y, dtype=np.float64)
    return np.arctan2(-v_y, -v_x)


def chaser_centred_bearing_degrees(x_mm, y_mm) -> np.ndarray:
    """``CHASER_CENTRED_BEARING`` of a point or displacement (float64 degrees)."""

    return np.degrees(np.arctan2(np.asarray(x_mm, dtype=np.float64), np.asarray(y_mm, dtype=np.float64)))


__all__ = [
    "ANGLE_CONVENTION",
    "CHASER_CENTRED_BEARING",
    "HANDEDNESS",
    "MEGABOUTS_HEAD_YAW",
    "chaser_centred_bearing_degrees",
    "heading_degrees",
    "image_angle_degrees",
    "left_axis",
    "megabouts_head_yaw_radians",
    "wrap_delta_degrees",
    "wrap_delta_radians",
]
