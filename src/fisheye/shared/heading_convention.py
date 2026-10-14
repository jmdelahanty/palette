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

Stimulus angles (agent-contracts ``citrus-palette/angles/
stimulus_angle_convention_v1.json``, PR 91, vendored as
``fisheye/shared/contracts/citrus_palette_stimulus_angle_convention_v1.json``):

``STIMULUS_ANGLE_CONVENTION`` (= ``ANGLE_CONVENTION``)
    The same id names the same rule on any y-down raster (stimulus canvas,
    camera view): 0 = raster right, +90 = raster up, counter-clockwise on the
    image, angle = ``degrees(atan2(-y, x))``. Recorded values are finite and
    not normalised (270, -90 and 630 name one direction); compare them only
    through ``wrap_delta_degrees``. A stimulus angle never crosses rasters by
    number: map its vector through the homography Jacobian instead.

``LEGACY_CANVAS_ANGLE`` (Citrus, before unified-H5 catalog ``angle_v1``)
    ``canvas_y_down_clockwise_legacy``: 0 = +x (right), +90 = down on the
    raster, positive clockwise on the image; vector ``(cos t, sin t)``, angle
    ``degrees(atan2(y, x))``. ``theta_v1 = -theta_legacy`` on the same raster
    (``legacy_canvas_angle_to_degrees``). Which convention a unified H5's
    stimulus angles follow is fixed by its admitted catalog binding
    (``fisheye.shared.unified_h5.schema.admitted_binding``), never by its date
    or by the session attribute alone.

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

# Stimulus direction angles share the heading's id (one rule for any y-down raster).
STIMULUS_ANGLE_CONVENTION = ANGLE_CONVENTION
LEGACY_CANVAS_ANGLE = "canvas_y_down_clockwise_legacy"

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


def legacy_canvas_angle_to_degrees(legacy_deg, *, dtype=np.float64) -> np.ndarray:
    """A ``LEGACY_CANVAS_ANGLE`` value in ``STIMULUS_ANGLE_CONVENTION`` degrees.

    Same raster, same vector, opposite sign: legacy 90 (down) is -90, legacy
    270 (up) is -270, i.e. 90. The result is not normalised (the current
    convention keeps finite values as recorded, and -180 stays -180); compare
    it only through ``wrap_delta_degrees``.
    """

    return -np.asarray(legacy_deg, dtype=dtype)


def stimulus_angle_vector(
    angle_deg, convention: str = STIMULUS_ANGLE_CONVENTION, *, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector ``(x, y)`` on a y-down raster named by a stimulus angle.

    ``convention`` is the one the recording's admitted binding fixed:
    ``STIMULUS_ANGLE_CONVENTION`` gives ``(cos t, -sin t)`` (+90 = up),
    ``LEGACY_CANVAS_ANGLE`` gives ``(cos t, sin t)`` (+90 = down).
    """

    if convention == STIMULUS_ANGLE_CONVENTION:
        sign = -1.0
    elif convention == LEGACY_CANVAS_ANGLE:
        sign = 1.0
    else:
        raise ValueError(f"unknown stimulus angle convention: {convention!r}")
    theta = np.radians(np.asarray(angle_deg, dtype=dtype))
    return np.cos(theta), sign * np.sin(theta)


def homography_jacobian(homography, point_xy) -> np.ndarray:
    """Local 2x2 Jacobian of a 3x3 homography at a source point.

    agent-contracts ``citrus-palette/angles`` ``plane.jacobian``: with
    ``w = h20 x + h21 y + h22`` and ``(u, v)`` the mapped point,
    ``J = [[(h00 - u h20)/w, (h01 - u h21)/w], [(h10 - v h20)/w, (h11 - v h21)/w]]``.
    """

    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("homography must be one finite 3x3 matrix")
    x, y = (float(value) for value in point_xy)
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if not np.isfinite(w) or w == 0.0:
        raise ValueError("homography maps the point to infinity")
    u = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
    v = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
    return np.array(
        [
            [(h[0, 0] - u * h[2, 0]) / w, (h[0, 1] - u * h[2, 1]) / w],
            [(h[1, 0] - v * h[2, 0]) / w, (h[1, 1] - v * h[2, 1]) / w],
        ],
        dtype=np.float64,
    )


def map_stimulus_direction(angle_deg, jacobian) -> tuple[float, int]:
    """Map a ``STIMULUS_ANGLE_CONVENTION`` direction into another y-down raster.

    The angle names a vector in the source raster; the vector (never the
    number) goes through the local Jacobian, and the result is the angle of
    the mapped vector in the destination raster. Returns ``(angle_deg,
    det_sign)``; ``det_sign == -1`` is a reflected rig (rotation sense
    reverses), as on omnifin0.
    """

    j = np.asarray(jacobian, dtype=np.float64)
    if j.shape != (2, 2) or not np.isfinite(j).all():
        raise ValueError("jacobian must be one finite 2x2 matrix")
    det = float(np.linalg.det(j))
    if det == 0.0:
        raise ValueError("jacobian is singular; the direction has no image")
    vx, vy = stimulus_angle_vector(angle_deg)
    mapped = j @ np.array([float(vx), float(vy)], dtype=np.float64)
    return float(heading_degrees(mapped[0], mapped[1])), (1 if det > 0.0 else -1)


def finite_difference_jacobian(point_map, point_xy, *, step_px: float = 0.5) -> np.ndarray:
    """Central-difference Jacobian of a point map ``(N, 2) -> (N, 2)``.

    The two-point form of ``homography_jacobian`` for a map known only as a
    point transform (for example a bound transform chain). Exact for affine
    maps; for a projective map the error is second order in ``step_px``.
    """

    x, y = (float(value) for value in point_xy)
    h = float(step_px)
    probes = np.array(
        [[x - h, y], [x + h, y], [x, y - h], [x, y + h]], dtype=np.float64
    )
    mapped = np.asarray(point_map(probes), dtype=np.float64)
    if mapped.shape != (4, 2) or not np.isfinite(mapped).all():
        raise ValueError("point map returned non-finite or misshapen points")
    return np.column_stack(
        [(mapped[1] - mapped[0]) / (2.0 * h), (mapped[3] - mapped[2]) / (2.0 * h)]
    )


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
    "LEGACY_CANVAS_ANGLE",
    "MEGABOUTS_HEAD_YAW",
    "STIMULUS_ANGLE_CONVENTION",
    "chaser_centred_bearing_degrees",
    "finite_difference_jacobian",
    "heading_degrees",
    "homography_jacobian",
    "image_angle_degrees",
    "left_axis",
    "legacy_canvas_angle_to_degrees",
    "map_stimulus_direction",
    "megabouts_head_yaw_radians",
    "stimulus_angle_vector",
    "wrap_delta_degrees",
    "wrap_delta_radians",
]
