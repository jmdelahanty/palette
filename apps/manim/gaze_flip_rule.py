"""Manim animation: ellipse-axis flip rule for larval zebrafish gaze.

Visualizes the geometry behind Palette's eye-gaze resolution and the silent
failure mode that motivates the major-axis-canonical refactor.

Scenes (in order):
    Scene1Setup        Fish, body frame (f-hat, l-hat), and the forward/backward
                       half-plane divider through the analyzed eye.
    Scene2Typical      The minor-axis flip rule resolving correctly through the
                       typical larval gaze range. The fitter's chosen endpoint
                       toggles to show the rule is invariant to that choice.
    Scene3Failure      The eye rotates past 90 deg from forward. The same flip
                       rule silently maps the resolved gaze to its diametric
                       opposite, with no warning.
    Scene4MajorAxis    Bianco / Engert-lab convention: resolve the 180 deg
                       ambiguity on the *major* (TN) axis, then derive gaze as
                       the perpendicular. The resolved gaze tracks ground truth
                       through the same rotation that broke Scene 3.
    Scene5Boundary     Boundary-jitter contrast: small fit noise near the half-
                       plane boundary causes the minor-axis-resolved gaze to
                       flicker by 180 deg, while the major-axis resolution is
                       stable.
    Scene6Vergence     Vergence math demo: both eyes converge symmetrically with
                       live Bianco-style nasal-rotation arithmetic and the
                       binocular-overlap zone shown as a green wedge.
    MasterCorrectScene The 'correct way' sequence: definitions, setup, major-
                       axis resolution, vergence. Skips failure-mode scenes.
    MasterScene        The full sequence (Scenes 1-6) including failure-mode
                       scenes; useful for the 'why we needed v5' story.

Render at standard 720p / 30fps:
    manim -pqm apps/manim/gaze_flip_rule.py Scene1Setup
    manim -pqm apps/manim/gaze_flip_rule.py Scene2Typical
    manim -pqm apps/manim/gaze_flip_rule.py Scene3Failure
    manim -pqm apps/manim/gaze_flip_rule.py Scene4MajorAxis
    manim -pqm apps/manim/gaze_flip_rule.py Scene5Boundary
    manim -pqm apps/manim/gaze_flip_rule.py Scene6Vergence
    manim -pqm apps/manim/gaze_flip_rule.py MasterCorrectScene
    manim -pqm apps/manim/gaze_flip_rule.py MasterScene
"""
from __future__ import annotations

import numpy as np
from manim import *


# ============================================================================
# Tunable parameters (tweak these to adjust styling, layout, or pacing)
# ============================================================================

# --- Colors ----------------------------------------------------------------
BODY_COLOR = "#3a3a3a"           # fish body outline + anatomical eye outlines
FRAME_COLOR = "#1f6feb"          # body-frame axes f-hat, l-hat
TRUTH_COLOR = "#27ae60"          # ground-truth gaze (green)
ERROR_COLOR = "#e74c3c"          # incorrectly resolved gaze (red)
RAW_COLOR = "#9aa0a6"            # raw ellipse-axis candidates (neutral gray)
RESOLVED_OK_COLOR = "#1f6feb"    # resolved g-hat when correct (blue)
MAJOR_AXIS_COLOR = "#ff8c00"     # major (TN) axis line + resolved arrow (orange)
HALFPLANE_COLOR = "#888888"      # dashed half-plane divider line
HALFPLANE_FILL_FORWARD = "#d8ecff"   # pale-blue fill for the forward half-plane
HALFPLANE_FILL_BACKWARD = "#ffdcdc"  # pale-red fill for the backward half-plane
LABEL_COLOR = "#1a1a1a"          # prose labels
BACKGROUND_COLOR = "#f7f7f7"     # off-white scene background

# --- Vision-cone palette (Scene 6 vergence demo) ----------------------------
# Match the standard Bianco/Easter & Nicola convention: pink-ish per eye, with
# a green binocular overlap that appears as the eyes converge.
LEFT_CONE_COLOR = "#f5b8b3"      # soft coral for the left eye's monocular field
RIGHT_CONE_COLOR = "#a3c8ff"     # soft blue for the right eye's monocular field
BINOCULAR_COLOR = "#9be59c"      # soft green for the binocular overlap

# Per-eye angular range. Bianco 2011 reports ~163 deg per eye in larval
# zebrafish, so half-cone angle is ~81.5 deg.
CONE_HALF_ANGLE_DEG = 81.5

# --- Body geometry (scene units; +y is forward / up the page) ---------------
# Eyes placed lower on the face so the larger ellipses fit on the broadened head.
EYE_LEFT_CENTER = np.array([-0.45, 0.95, 0.0])
EYE_RIGHT_CENTER = np.array([0.45, 0.95, 0.0])
SWIM_BLADDER = np.array([0.0, -0.30, 0.0])
F_HAT = np.array([0.0, 1.0, 0.0])
L_HAT = np.array([-1.0, 0.0, 0.0])
F_HAT_LENGTH = 1.55
L_HAT_LENGTH = 1.20

# Stylized larval-zebrafish body polygon traced CCW from the head tip.
# Head is broadened (rounded top, widest near the eye line); the tail has an
# extended near-parallel "stem" before it tapers to a point, matching larval
# anatomy where the post-anal tail is roughly half the total length.
FISH_BODY_POINTS = [
    [0.00, 1.70, 0.0],
    [0.35, 1.62, 0.0],
    [0.62, 1.40, 0.0],
    [0.78, 1.10, 0.0],
    [0.72, 0.65, 0.0],
    [0.50, 0.15, 0.0],
    [0.32, -0.35, 0.0],
    [0.20, -1.30, 0.0],
    [0.16, -2.40, 0.0],
    [0.12, -3.30, 0.0],
    [0.00, -3.95, 0.0],
    [-0.12, -3.30, 0.0],
    [-0.16, -2.40, 0.0],
    [-0.20, -1.30, 0.0],
    [-0.32, -0.35, 0.0],
    [-0.50, 0.15, 0.0],
    [-0.72, 0.65, 0.0],
    [-0.78, 1.10, 0.0],
    [-0.62, 1.40, 0.0],
    [-0.35, 1.62, 0.0],
]

# --- Fitted-ellipse shape (pure visual; not derived from real data) ---------
# Major axis is perpendicular to gaze; minor axis IS the gaze direction.
# Aspect ratio ~2.8 reads clearly as elliptical at scene scale.
ELLIPSE_MAJOR = 0.78
ELLIPSE_MINOR = 0.28
GAZE_LINE_LENGTH = 1.55    # full minor-axis line shown across the ellipse
GAZE_ARROW_LENGTH = 1.10   # length of resolved/truth/derived gaze arrows
MAJOR_LINE_LENGTH = 1.55   # full major-axis line shown across the ellipse

# --- Half-plane divider (drawn through the analyzed eye) -------------------
HALFPLANE_X_LEFT = -3.6
HALFPLANE_X_RIGHT = 3.6
HALFPLANE_Y = EYE_LEFT_CENTER[1]

# --- Animation timing (seconds) --------------------------------------------
SETUP_HOLD = 2.0
TYPICAL_RUN_TIME = 5.0
TYPICAL_TOGGLE_PAUSE = 0.55
FAILURE_RUN_TIME = 6.0
FAILURE_HOLD = 3.0
MAJOR_REWIND_TIME = 1.5
MAJOR_RUN_TIME = 6.0
MAJOR_HOLD = 2.5
BOUNDARY_FLICKER_HALF = 0.18

# --- Rotation key angles (radians, CCW from +y forward) --------------------
# theta is the angle of the minor (gaze) axis from forward. The major axis is
# perpendicular: at theta=90 deg the gaze points fully lateral and the major
# axis is along the body AP -- i.e., the larval-eye default (resting) state.
#
# Biologically realistic per-eye rotation extents (Bianco et al. 2011):
#   - Nasal convergence: 0 deg - 50 deg from rest -> theta in [40 deg, 90 deg].
#   - Past-lateral divergence: less common, plausible up to ~35 deg past lateral
#     -> theta in [90 deg, 125 deg].
# We sweep within these ranges so the animation reflects motion the fish
# actually executes; the failure mode is still unambiguous because it kicks in
# at the exact moment the gaze crosses 90 deg.
THETA_TYPICAL_START = np.deg2rad(55.0)   # ~35 deg nasal: comfortable mid-convergence
THETA_TYPICAL_END = np.deg2rad(85.0)     # near rest, just inside forward half-plane
THETA_FAILURE_END = np.deg2rad(125.0)    # ~35 deg past lateral: extreme but biological
THETA_MAJOR_REWIND = np.deg2rad(55.0)
THETA_MAJOR_END = np.deg2rad(125.0)      # same range as failure scene
THETA_BOUNDARY_LOW = np.deg2rad(85.0)    # near the half-plane boundary
THETA_BOUNDARY_HIGH = np.deg2rad(95.0)


# ============================================================================
# Pure-geometry helpers
# ============================================================================

def gaze_unit(theta: float) -> np.ndarray:
    """Unit vector along the minor (gaze) axis at angle theta CCW from +y."""
    return np.array([-np.sin(theta), np.cos(theta), 0.0])


def major_unit(theta: float) -> np.ndarray:
    """Unit vector along the major (TN) axis -- perpendicular to gaze."""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def in_forward_half(v: np.ndarray) -> bool:
    """True iff v lies in the forward half-plane (v . f-hat >= 0)."""
    return float(np.dot(v, F_HAT)) >= 0.0


def apply_flip_rule(raw_dir: np.ndarray) -> np.ndarray:
    """Forward half-plane rule: negate raw_dir if it sits in the backward half."""
    return raw_dir if in_forward_half(raw_dir) else -raw_dir


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def perpendicular_left(v: np.ndarray) -> np.ndarray:
    """+90 deg CCW rotation of a 2D-in-xy vector (the anatomical-left perpendicular).

    For the LEFT eye, this is the convention that maps the resolved major axis
    onto the outward-pointing gaze direction.
    """
    return np.array([-v[1], v[0], 0.0])


# ============================================================================
# Base scene with shared object builders + scripted phases
# ============================================================================

class GazeBase(Scene):
    """Shared infrastructure for all gaze scenes.

    Each per-scene class calls build_objects() once in construct(), then the
    scripted play_*() methods. MasterScene chains them all.
    """

    # ---- mobject construction -------------------------------------------

    def build_objects(self) -> None:
        self.camera.background_color = BACKGROUND_COLOR

        # ----- static fish anatomy --------------------------------------
        self.fish_body = Polygon(
            *FISH_BODY_POINTS, color=BODY_COLOR, stroke_width=2.5,
        )
        self.fish_body.set_fill(BODY_COLOR, opacity=0.06)

        # Static anatomical eyes match the fit ellipse at default lateral gaze:
        # width = ELLIPSE_MINOR, height = ELLIPSE_MAJOR — so the visual size
        # doesn't change when Scene 2 swaps the static left eye for the rotating
        # fit ellipse.
        self.eye_left = Ellipse(
            width=ELLIPSE_MINOR, height=ELLIPSE_MAJOR,
            color=BODY_COLOR, stroke_width=2.0,
        ).move_to(EYE_LEFT_CENTER).set_fill(BODY_COLOR, opacity=0.10)

        self.eye_right = Ellipse(
            width=ELLIPSE_MINOR, height=ELLIPSE_MAJOR,
            color=BODY_COLOR, stroke_width=2.0,
        ).move_to(EYE_RIGHT_CENTER).set_fill(BODY_COLOR, opacity=0.10)

        self.bladder = Dot(point=SWIM_BLADDER, color=BODY_COLOR, radius=0.07)
        self.bladder_label = Text(
            "swim bladder", font_size=18, color=BODY_COLOR,
        ).next_to(self.bladder, RIGHT, buff=0.55)

        # ----- body-frame axes ------------------------------------------
        f_tip = SWIM_BLADDER + F_HAT_LENGTH * F_HAT
        self.f_arrow = Arrow(
            start=SWIM_BLADDER, end=f_tip, color=FRAME_COLOR,
            buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.13,
        )
        # f̂, l̂ use the unicode combining circumflex (U+0302) so we don't need LaTeX.
        self.f_label = Text("f̂", color=FRAME_COLOR, font_size=36, weight=BOLD)
        self.f_label.next_to(f_tip, RIGHT, buff=0.10)

        l_tip = SWIM_BLADDER + L_HAT_LENGTH * L_HAT
        self.l_arrow = Arrow(
            start=SWIM_BLADDER, end=l_tip, color=FRAME_COLOR,
            buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.13,
        )
        self.l_label = Text("l̂", color=FRAME_COLOR, font_size=36, weight=BOLD)
        self.l_label.next_to(l_tip, DOWN, buff=0.10)

        # ----- half-plane shading (drawn behind everything via z_index) ---
        # Polygons extend past the visible frame so we don't see edges.
        shade_x = 8.0
        shade_top = 4.5
        shade_bot = -4.5
        self.forward_half_shade = Polygon(
            [-shade_x, HALFPLANE_Y, 0.0],
            [shade_x, HALFPLANE_Y, 0.0],
            [shade_x, shade_top, 0.0],
            [-shade_x, shade_top, 0.0],
            color=HALFPLANE_FILL_FORWARD, stroke_width=0,
        ).set_fill(HALFPLANE_FILL_FORWARD, opacity=1.0)
        self.forward_half_shade.set_z_index(-10)

        self.backward_half_shade = Polygon(
            [-shade_x, shade_bot, 0.0],
            [shade_x, shade_bot, 0.0],
            [shade_x, HALFPLANE_Y, 0.0],
            [-shade_x, HALFPLANE_Y, 0.0],
            color=HALFPLANE_FILL_BACKWARD, stroke_width=0,
        ).set_fill(HALFPLANE_FILL_BACKWARD, opacity=1.0)
        self.backward_half_shade.set_z_index(-10)

        # ----- half-plane divider through the analyzed (left) eye -------
        self.halfplane_line = DashedLine(
            start=np.array([HALFPLANE_X_LEFT, HALFPLANE_Y, 0.0]),
            end=np.array([HALFPLANE_X_RIGHT, HALFPLANE_Y, 0.0]),
            color=HALFPLANE_COLOR, stroke_width=1.5, dash_length=0.10,
        )
        self.fwd_label = Text(
            "forward half-plane:  f̂ · v  >  0",
            color=LABEL_COLOR, font_size=20,
        ).move_to(np.array([3.0, HALFPLANE_Y + 0.45, 0.0]))
        self.bwd_label = Text(
            "backward half-plane:  f̂ · v  <  0",
            color=LABEL_COLOR, font_size=20,
        ).move_to(np.array([3.0, HALFPLANE_Y - 0.45, 0.0]))

        # ----- trackers -------------------------------------------------
        # theta is the gaze (minor-axis) angle CCW from forward.
        # fitter_sign simulates which of the two endpoints OpenCV happens to
        # return on a given frame (+1 or -1 are equally valid raw outputs).
        # convergence_tracker is the Scene 6 per-eye nasal rotation magnitude
        # (in degrees), shared symmetrically by both eyes.
        self.theta_tracker = ValueTracker(THETA_TYPICAL_START)
        self.fitter_sign_tracker = ValueTracker(1.0)
        self.convergence_tracker = ValueTracker(0.0)
        # Controls the cones' fill opacity in Scene 6 (0 = invisible, 1 = full).
        # Animated 0 -> 1 to make the cones gradually reveal rather than pop.
        self.verg_cone_alpha = ValueTracker(0.0)

        # ----- dynamic (tracker-driven) mobjects ------------------------
        self.fit_ellipse = always_redraw(self._make_fit_ellipse)
        self.minor_axis_line = always_redraw(self._make_minor_axis_line)
        self.raw_pos_arrow = always_redraw(lambda: self._make_raw_arrow(+1))
        self.raw_neg_arrow = always_redraw(lambda: self._make_raw_arrow(-1))
        self.fitter_marker = always_redraw(self._make_fitter_marker)
        self.resolved_arrow = always_redraw(self._make_minor_resolved_arrow)
        self.truth_arrow = always_redraw(self._make_truth_arrow)
        self.error_indicator = always_redraw(self._make_error_indicator)

        # Major-axis (Scene 4) mobjects
        self.major_axis_line = always_redraw(self._make_major_axis_line)
        self.major_resolved_arrow = always_redraw(self._make_major_resolved_arrow)
        self.tn_endpoints = always_redraw(self._make_tn_endpoints)
        self.derived_gaze_arrow = always_redraw(self._make_derived_gaze_arrow)
        self.major_dot_label = always_redraw(self._make_major_dot_label)

        # Vergence-demo (Scene 6) mobjects: both eyes rotate symmetrically.
        self.verg_eye_left = always_redraw(lambda: self._make_verg_eye(0))
        self.verg_eye_right = always_redraw(lambda: self._make_verg_eye(1))
        self.verg_left_major = always_redraw(lambda: self._make_verg_major_line(0))
        self.verg_right_major = always_redraw(lambda: self._make_verg_major_line(1))
        self.verg_left_tn = always_redraw(lambda: self._make_verg_tn(0))
        self.verg_right_tn = always_redraw(lambda: self._make_verg_tn(1))
        self.verg_readout = always_redraw(self._make_verg_readout)
        self.verg_left_cone = always_redraw(lambda: self._make_verg_cone(0))
        self.verg_right_cone = always_redraw(lambda: self._make_verg_cone(1))
        self.verg_binocular = always_redraw(self._make_binocular_zone)

    # ---- dynamic mobject factories --------------------------------------

    def _make_fit_ellipse(self) -> Mobject:
        theta = self.theta_tracker.get_value()
        return (
            Ellipse(
                width=ELLIPSE_MAJOR, height=ELLIPSE_MINOR,
                color=RAW_COLOR, stroke_width=2.0,
            )
            .set_fill(RAW_COLOR, opacity=0.10)
            .move_to(EYE_LEFT_CENTER)
            .rotate(theta)
        )

    def _make_minor_axis_line(self) -> Mobject:
        """Undirected minor-axis line through the ellipse (no arrowhead)."""
        theta = self.theta_tracker.get_value()
        g = gaze_unit(theta)
        half = 0.5 * GAZE_LINE_LENGTH
        return Line(
            start=EYE_LEFT_CENTER - half * g,
            end=EYE_LEFT_CENTER + half * g,
            color=RAW_COLOR, stroke_width=3,
        )

    def _make_raw_arrow(self, sign: int) -> Mobject:
        """One of the two faint candidate arrows (the line has two endpoints)."""
        theta = self.theta_tracker.get_value()
        g = gaze_unit(theta)
        end = EYE_LEFT_CENTER + sign * GAZE_ARROW_LENGTH * g
        return Arrow(
            start=EYE_LEFT_CENTER, end=end, color=RAW_COLOR, buff=0,
            stroke_width=2, stroke_opacity=0.55,
            max_tip_length_to_length_ratio=0.13,
        )

    def _make_fitter_marker(self) -> Mobject:
        """Small ring marking which endpoint the 'fitter' currently returned."""
        sign = self.fitter_sign_tracker.get_value()
        theta = self.theta_tracker.get_value()
        g = gaze_unit(theta)
        pos = EYE_LEFT_CENTER + sign * 0.5 * GAZE_LINE_LENGTH * g
        ring = Annulus(
            inner_radius=0.10, outer_radius=0.14,
            color=RAW_COLOR, fill_opacity=0.6,
        ).move_to(pos)
        tag = Text("raw", font_size=14, color=RAW_COLOR).next_to(
            ring, UP, buff=0.05,
        )
        return VGroup(ring, tag)

    def _make_minor_resolved_arrow(self) -> Mobject:
        """Resolved g-hat from the minor-axis flip rule. Color reflects correctness."""
        sign = self.fitter_sign_tracker.get_value()
        theta = self.theta_tracker.get_value()
        raw = sign * gaze_unit(theta)
        resolved = apply_flip_rule(raw)
        truth = gaze_unit(theta)
        is_correct = bool(np.dot(resolved, truth) > 0.0)
        color = RESOLVED_OK_COLOR if is_correct else ERROR_COLOR
        arrow = Arrow(
            start=EYE_LEFT_CENTER,
            end=EYE_LEFT_CENTER + GAZE_ARROW_LENGTH * resolved,
            color=color, buff=0, stroke_width=5,
            max_tip_length_to_length_ratio=0.18,
        )
        # Pin the ĝ label above the arrow tip (not in the direction the arrow
        # points) so it doesn't stack on the "truth" label when both arrows
        # happen to point the same way (early Scene 3, before the eye crosses
        # the half-plane boundary).
        label = Text("ĝ", color=color, font_size=28, weight=BOLD)
        label.next_to(arrow.get_end(), UP, buff=0.10)
        return VGroup(arrow, label)

    def _make_truth_arrow(self) -> Mobject:
        """Ground-truth gaze: always the actual physical gaze direction (green)."""
        theta = self.theta_tracker.get_value()
        truth = gaze_unit(theta)
        arrow = Arrow(
            start=EYE_LEFT_CENTER,
            end=EYE_LEFT_CENTER + GAZE_ARROW_LENGTH * truth,
            color=TRUTH_COLOR, buff=0, stroke_width=5,
            max_tip_length_to_length_ratio=0.18,
        )
        label = Text("truth", font_size=20, color=TRUTH_COLOR)
        label.next_to(arrow.get_end(), normalize(truth), buff=0.12)
        return VGroup(arrow, label)

    def _make_error_indicator(self) -> Mobject:
        """Curved arrow + '180 deg error' label, only when resolved disagrees with truth."""
        theta = self.theta_tracker.get_value()
        sign = self.fitter_sign_tracker.get_value()
        truth = gaze_unit(theta)
        raw = sign * truth
        resolved = apply_flip_rule(raw)
        if np.dot(truth, resolved) > 0.0:
            return VGroup()  # no error in the forward half-plane
        truth_tip = EYE_LEFT_CENTER + GAZE_ARROW_LENGTH * truth
        resolved_tip = EYE_LEFT_CENTER + GAZE_ARROW_LENGTH * resolved
        arc = ArcBetweenPoints(
            start=truth_tip, end=resolved_tip,
            angle=PI * 0.6, color=ERROR_COLOR, stroke_width=3,
        )
        # Position the label below the fish body so it doesn't collide with
        # the swim-bladder text or the bottom-edge narration.
        label = Text(
            "180 deg error", font_size=24, color=ERROR_COLOR, weight=BOLD,
        ).move_to(np.array([3.0, -2.2, 0.0]))
        return VGroup(arc, label)

    # ---- Scene 4 (major-axis) mobject factories -------------------------

    def _make_major_axis_line(self) -> Mobject:
        """Solid orange line along the major (TN) axis."""
        theta = self.theta_tracker.get_value()
        m = major_unit(theta)
        half = 0.5 * MAJOR_LINE_LENGTH
        return Line(
            start=EYE_LEFT_CENTER - half * m,
            end=EYE_LEFT_CENTER + half * m,
            color=MAJOR_AXIS_COLOR, stroke_width=4,
        )

    def _make_major_resolved_arrow(self) -> Mobject:
        """Resolved major axis (after forward half-plane rule): rostral endpoint."""
        theta = self.theta_tracker.get_value()
        resolved_major = apply_flip_rule(major_unit(theta))
        # End the arrow at the line endpoint so the T/N labels (added separately)
        # frame the line ends cleanly without an arrowhead poking past N.
        arrow_length = 0.5 * MAJOR_LINE_LENGTH
        arrow = Arrow(
            start=EYE_LEFT_CENTER,
            end=EYE_LEFT_CENTER + arrow_length * resolved_major,
            color=MAJOR_AXIS_COLOR, buff=0, stroke_width=4,
            max_tip_length_to_length_ratio=0.20,
        )
        return arrow

    def _make_tn_endpoints(self) -> Mobject:
        """T (temporal) / N (nasal) labels at the two ends of the resolved major axis."""
        theta = self.theta_tracker.get_value()
        resolved_major = apply_flip_rule(major_unit(theta))
        offset = 0.5 * MAJOR_LINE_LENGTH + 0.30  # past line endpoint with buffer
        n_label = Text(
            "N", font_size=22, color=MAJOR_AXIS_COLOR, weight=BOLD,
        ).move_to(EYE_LEFT_CENTER + offset * resolved_major)
        t_label = Text(
            "T", font_size=22, color=MAJOR_AXIS_COLOR, weight=BOLD,
        ).move_to(EYE_LEFT_CENTER - offset * resolved_major)
        return VGroup(t_label, n_label)

    def _make_derived_gaze_arrow(self) -> Mobject:
        """Gaze derived as +90 deg rotation of the resolved major (left-eye outward)."""
        theta = self.theta_tracker.get_value()
        resolved_major = apply_flip_rule(major_unit(theta))
        derived = perpendicular_left(resolved_major)
        truth = gaze_unit(theta)
        is_correct = bool(np.dot(derived, truth) > 0.0)
        color = TRUTH_COLOR if is_correct else ERROR_COLOR
        arrow = Arrow(
            start=EYE_LEFT_CENTER,
            end=EYE_LEFT_CENTER + GAZE_ARROW_LENGTH * derived,
            color=color, buff=0, stroke_width=5,
            max_tip_length_to_length_ratio=0.18,
        )
        label = Text("ĝ derived", color=color, font_size=22, weight=BOLD)
        label.next_to(arrow.get_end(), normalize(derived), buff=0.12)
        return VGroup(arrow, label)

    def _make_major_dot_label(self) -> Mobject:
        """Live readout of |dot(major, f-hat)| -- shows the half-plane test margin."""
        theta = self.theta_tracker.get_value()
        d = float(np.dot(major_unit(theta), F_HAT))
        text = Text(
            f"|m̂ · f̂|  =  {abs(d):.2f}",
            color=MAJOR_AXIS_COLOR, font_size=22,
        ).move_to(np.array([3.2, -1.7, 0.0]))
        return text

    # ---- Scene 6 (vergence demo) factories ------------------------------

    def _verg_theta(self, eye_idx: int) -> float:
        """Per-eye theta (CCW from +y) given the symmetric convergence magnitude."""
        alpha_rad = np.deg2rad(self.convergence_tracker.get_value())
        if eye_idx == 0:
            return np.pi / 2 - alpha_rad   # left eye: 90 deg at rest, decreases nasally
        return -np.pi / 2 + alpha_rad      # right eye: -90 deg at rest, increases nasally

    def _make_verg_eye(self, eye_idx: int) -> Mobject:
        """Rotating fit ellipse for the left (0) or right (1) eye."""
        theta = self._verg_theta(eye_idx)
        center = EYE_LEFT_CENTER if eye_idx == 0 else EYE_RIGHT_CENTER
        return (
            Ellipse(
                width=ELLIPSE_MAJOR, height=ELLIPSE_MINOR,
                color=BODY_COLOR, stroke_width=2.0,
            )
            .set_fill(BODY_COLOR, opacity=0.10)
            .move_to(center)
            .rotate(theta)
        )

    def _make_verg_major_line(self, eye_idx: int) -> Mobject:
        """Orange TN-axis line through the eye's center."""
        theta = self._verg_theta(eye_idx)
        center = EYE_LEFT_CENTER if eye_idx == 0 else EYE_RIGHT_CENTER
        direction = major_unit(theta)
        half = 0.5 * MAJOR_LINE_LENGTH
        return Line(
            start=center - half * direction,
            end=center + half * direction,
            color=MAJOR_AXIS_COLOR, stroke_width=4,
        )

    def _make_verg_tn(self, eye_idx: int) -> Mobject:
        """T (temporal) / N (nasal) labels at the line endpoints."""
        theta = self._verg_theta(eye_idx)
        center = EYE_LEFT_CENTER if eye_idx == 0 else EYE_RIGHT_CENTER
        resolved = apply_flip_rule(major_unit(theta))
        offset = 0.5 * MAJOR_LINE_LENGTH + 0.30
        n_label = Text(
            "N", font_size=20, color=MAJOR_AXIS_COLOR, weight=BOLD,
        ).move_to(center + offset * resolved)
        t_label = Text(
            "T", font_size=20, color=MAJOR_AXIS_COLOR, weight=BOLD,
        ).move_to(center - offset * resolved)
        return VGroup(t_label, n_label)

    def _make_verg_readout(self) -> Mobject:
        """Live-updating Bianco eye-frame vergence math, anchored on the left side."""
        # Round once and use the rounded values everywhere so the displayed
        # arithmetic is internally consistent (otherwise +1.0 + +1.0 can show
        # as 1.9 due to underlying float rounding).
        left_disp = round(self.convergence_tracker.get_value(), 1)
        right_disp = left_disp
        verg_disp = round(left_disp + right_disp, 1)

        per_eye = VGroup(
            Text(
                f"left eye nasal:    {left_disp:+.1f}°",
                font_size=22, color=MAJOR_AXIS_COLOR,
            ),
            Text(
                f"right eye nasal:  {right_disp:+.1f}°",
                font_size=22, color=MAJOR_AXIS_COLOR,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        verg = VGroup(
            Text(
                "vergence  =  nasal_L  +  nasal_R",
                font_size=22, color=LABEL_COLOR,
            ),
            Text(
                f"            =  {left_disp:+.1f}°  +  {right_disp:+.1f}°",
                font_size=22, color=LABEL_COLOR,
            ),
            Text(
                f"            =  {verg_disp:.1f}°",
                font_size=26, color=LABEL_COLOR, weight=BOLD,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        block = VGroup(per_eye, verg).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        block.to_edge(LEFT, buff=0.5)
        return block

    def _make_verg_cone(self, eye_idx: int) -> Mobject:
        """Translucent monocular vision cone (sector) for one eye."""
        alpha_rad = np.deg2rad(self.convergence_tracker.get_value())
        half = np.deg2rad(CONE_HALF_ANGLE_DEG)
        if eye_idx == 0:
            bisector = np.pi - alpha_rad      # left eye gaze in math conv
            apex = EYE_LEFT_CENTER
            cone_color = LEFT_CONE_COLOR
        else:
            bisector = alpha_rad              # right eye gaze in math conv
            apex = EYE_RIGHT_CENTER
            cone_color = RIGHT_CONE_COLOR
        # Annular sector with inner_radius=0 reduces to a pie slice; its apex
        # sits at the origin, so we shift it to the eye center.
        sector = AnnularSector(
            inner_radius=0.0,
            outer_radius=8.0,
            angle=2.0 * half,
            start_angle=bisector - half,
            color=cone_color,
            fill_opacity=0.22 * float(self.verg_cone_alpha.get_value()),
            stroke_width=0,
        ).shift(apex)
        sector.set_z_index(-3)
        return sector

    def _make_binocular_zone(self) -> Mobject:
        """Triangular polygon for the binocular intersection of the two cones.

        Returns an empty VGroup when the cones don't yet overlap (small alpha)
        or when the inner-boundary rays are parallel.
        """
        alpha_rad = np.deg2rad(self.convergence_tracker.get_value())
        half = np.deg2rad(CONE_HALF_ANGLE_DEG)

        # Inner-boundary rays of each cone (the side closest to the midline).
        # These rays delimit the binocular zone in front of the fish.
        L_inner_angle = (np.pi - alpha_rad) - half  # left cone, CW edge
        R_inner_angle = alpha_rad + half             # right cone, CCW edge
        L_dir = np.array([np.cos(L_inner_angle), np.sin(L_inner_angle), 0.0])
        R_dir = np.array([np.cos(R_inner_angle), np.sin(R_inner_angle), 0.0])

        L_apex = EYE_LEFT_CENTER
        R_apex = EYE_RIGHT_CENTER

        # Solve L_apex + t1 * L_dir = R_apex + t2 * R_dir for (t1, t2) -- the
        # ray-ray intersection in 2D.
        det = L_dir[0] * (-R_dir[1]) - (-R_dir[0]) * L_dir[1]
        if abs(det) < 1e-6:
            return VGroup()
        b = R_apex - L_apex
        t1 = (b[0] * (-R_dir[1]) - (-R_dir[0]) * b[1]) / det
        t2 = (L_dir[0] * b[1] - L_dir[1] * b[0]) / det
        if t1 <= 0 or t2 <= 0:
            return VGroup()  # rays meet behind the eyes (no forward overlap)

        meeting = L_apex + t1 * L_dir
        far_dist = 6.0
        far_along_L = meeting + far_dist * L_dir
        far_along_R = meeting + far_dist * R_dir

        polygon = Polygon(
            meeting, far_along_L, far_along_R,
            color=BINOCULAR_COLOR,
            fill_opacity=0.45 * float(self.verg_cone_alpha.get_value()),
            stroke_width=0,
        )
        polygon.set_z_index(-2)
        return polygon

    # ---- scripted phases ------------------------------------------------

    def play_definitions(self) -> None:
        """Title card: define f-hat, l-hat, and the TN axis before the fish appears."""
        self.camera.background_color = BACKGROUND_COLOR

        title = Text(
            "Conventions used in this animation:",
            font_size=30, color=LABEL_COLOR,
        )

        # f-hat: large symbol + a short upward arrow next to it for direction cue
        f_arrow = Arrow(
            start=ORIGIN, end=UP * 0.55, color=FRAME_COLOR,
            buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.30,
        )
        f_label = Text("f̂", font_size=52, color=FRAME_COLOR, weight=BOLD)
        f_icon = VGroup(f_arrow, f_label).arrange(RIGHT, buff=0.18)
        f_text = Text(
            "forward (rostral) — direction the fish is heading",
            font_size=24, color=LABEL_COLOR,
        )
        f_row = VGroup(f_icon, f_text).arrange(RIGHT, buff=0.6, aligned_edge=DOWN)

        # l-hat: large symbol + a short leftward arrow next to it for direction cue
        l_arrow = Arrow(
            start=ORIGIN, end=LEFT * 0.55, color=FRAME_COLOR,
            buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.30,
        )
        l_label = Text("l̂", font_size=52, color=FRAME_COLOR, weight=BOLD)
        l_icon = VGroup(l_arrow, l_label).arrange(RIGHT, buff=0.18)
        l_text = Text(
            "anatomical left — perpendicular to f̂, fish's left side",
            font_size=24, color=LABEL_COLOR,
        )
        l_row = VGroup(l_icon, l_text).arrange(RIGHT, buff=0.6, aligned_edge=DOWN)

        # TN axis icon: a short orange line with T and N at the two ends
        tn_t = Text("T", font_size=28, color=MAJOR_AXIS_COLOR, weight=BOLD)
        tn_n = Text("N", font_size=28, color=MAJOR_AXIS_COLOR, weight=BOLD)
        tn_line = Line(
            start=LEFT * 0.40, end=RIGHT * 0.40,
            color=MAJOR_AXIS_COLOR, stroke_width=4,
        )
        tn_t.next_to(tn_line.get_start(), LEFT, buff=0.08)
        tn_n.next_to(tn_line.get_end(), RIGHT, buff=0.08)
        tn_icon = VGroup(tn_t, tn_line, tn_n)
        tn_text = Text(
            "temporo-nasal eye axis — the eye's anatomical long axis",
            font_size=24, color=LABEL_COLOR,
        )
        tn_row = VGroup(tn_icon, tn_text).arrange(RIGHT, buff=0.6, aligned_edge=DOWN)

        rows = VGroup(f_row, l_row, tn_row).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        block = VGroup(title, rows).arrange(DOWN, buff=0.7).move_to(ORIGIN)

        self.play(FadeIn(title), run_time=0.5)
        self.play(FadeIn(rows), run_time=0.7)
        self.wait(4.5)
        self.play(FadeOut(block), run_time=0.6)

    def play_setup(self, animated: bool = True) -> None:
        """Scene 1: introduce fish, body frame, and half-plane divider."""
        if animated:
            # Phase 1: fish anatomy
            self.play(
                Create(self.fish_body),
                FadeIn(self.eye_left), FadeIn(self.eye_right),
                FadeIn(self.bladder), Write(self.bladder_label),
                run_time=1.4,
            )
            # Phase 2: derive heading (f-hat) from swim bladder -> eye midpoint
            self._play_heading_derivation()
            # Phase 3: anatomical-left axis (perpendicular to heading)
            self.play(
                GrowArrow(self.l_arrow), Write(self.l_label),
                run_time=0.8,
            )
            # Phase 4: half-plane
            self.play(
                Create(self.halfplane_line),
                FadeIn(self.forward_half_shade),
                FadeIn(self.backward_half_shade),
                FadeIn(self.fwd_label), FadeIn(self.bwd_label),
                run_time=1.0,
            )
        else:
            self.add(
                self.forward_half_shade, self.backward_half_shade,
                self.fish_body, self.eye_left, self.eye_right,
                self.bladder, self.bladder_label,
                self.f_arrow, self.f_label, self.l_arrow, self.l_label,
                self.halfplane_line, self.fwd_label, self.bwd_label,
            )
        self.wait(SETUP_HOLD)

    def _play_heading_derivation(self) -> None:
        """Show that the body forward axis f-hat is the swim-bladder-to-eye-midpoint vector."""
        midpoint = (EYE_LEFT_CENTER + EYE_RIGHT_CENTER) / 2.0

        # Just a marker dot at the midpoint; the bottom caption explains what
        # it represents (a label here would overlap the eye outlines).
        midpoint_dot = Dot(point=midpoint, color=FRAME_COLOR, radius=0.08)
        midpoint_dot.set_z_index(2)

        # Dashed construction line: swim bladder -> midpoint
        construction = DashedLine(
            start=SWIM_BLADDER, end=midpoint,
            color=FRAME_COLOR, stroke_width=3.0, dash_length=0.12,
        )

        caption = Text(
            "f̂ points from the swim bladder to the midpoint between the eyes",
            font_size=20, color=LABEL_COLOR,
        ).to_edge(DOWN, buff=0.4)

        # Mark the midpoint
        self.play(FadeIn(midpoint_dot), run_time=0.3)
        # Draw the construction line + caption
        self.play(Create(construction), FadeIn(caption), run_time=0.8)
        self.wait(1.8)
        # Replace the construction with the canonical f-hat arrow (which
        # extends slightly past the eye midpoint for visual clarity). The
        # "swim bladder" text label has done its job once heading is defined,
        # so we fade it here too (the dot stays as a reference point).
        self.play(
            FadeOut(construction),
            FadeOut(midpoint_dot),
            FadeOut(self.bladder_label),
            GrowArrow(self.f_arrow),
            Write(self.f_label),
            FadeOut(caption),
            run_time=0.8,
        )

    def _ensure_minor_dynamics(self) -> None:
        """Add fit ellipse + minor-axis dynamic mobjects if not already added.

        Also fades out the static anatomical left eye, since the fit ellipse
        replaces it as the analyzed object.
        """
        if self.fit_ellipse not in self.mobjects:
            if self.eye_left in self.mobjects:
                self.play(FadeOut(self.eye_left), run_time=0.4)
            self.add(
                self.fit_ellipse, self.minor_axis_line,
                self.raw_pos_arrow, self.raw_neg_arrow,
                self.fitter_marker, self.resolved_arrow,
            )

    def play_typical(self) -> None:
        """Scene 2: the minor-axis flip rule resolving correctly."""
        self.theta_tracker.set_value(THETA_TYPICAL_START)
        self.fitter_sign_tracker.set_value(1.0)
        self._ensure_minor_dynamics()

        narration = Text(
            "Flip rule on the minor (gaze) axis: keep the endpoint in the forward half-plane.",
            font_size=22, color=LABEL_COLOR,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(narration), run_time=0.5)
        self.wait(0.4)

        # Sweep the typical larval gaze range (eye in forward half-plane)
        self.play(
            self.theta_tracker.animate.set_value(THETA_TYPICAL_END),
            run_time=TYPICAL_RUN_TIME, rate_func=smooth,
        )

        invariance = Text(
            "Resolved gaze is invariant to which endpoint the fitter happens to return.",
            font_size=20, color=LABEL_COLOR,
        ).next_to(narration, UP, buff=0.10)
        self.play(FadeIn(invariance), run_time=0.4)

        # Toggle the simulated fitter sign a few times -- resolved arrow should
        # not move, demonstrating the rule's invariance.
        for _ in range(3):
            self.fitter_sign_tracker.set_value(-1.0)
            self.wait(TYPICAL_TOGGLE_PAUSE)
            self.fitter_sign_tracker.set_value(1.0)
            self.wait(TYPICAL_TOGGLE_PAUSE)

        self.wait(0.6)
        self.play(FadeOut(narration), FadeOut(invariance), run_time=0.5)

    def play_failure(self) -> None:
        """Scene 3: rotate past 90 deg; the rule produces a 180-degree error."""
        self._ensure_minor_dynamics()
        self.add(self.truth_arrow, self.error_indicator)

        narration = Text(
            "When the eye rotates past 90 deg from forward, the rule silently flips gaze 180 deg.",
            font_size=22, color=LABEL_COLOR,
        ).to_edge(DOWN, buff=0.4)
        # Legend at upper-left, well clear of the bottom-edge narration.
        legend = VGroup(
            Text("ground truth", font_size=18, color=TRUTH_COLOR),
            Text("resolved by rule", font_size=18, color=ERROR_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_corner(UL, buff=0.6)

        self.play(FadeIn(narration), FadeIn(legend), run_time=0.6)

        # Continue rotating past 90 deg, deep into the backward half-plane
        self.play(
            self.theta_tracker.animate.set_value(THETA_FAILURE_END),
            run_time=FAILURE_RUN_TIME, rate_func=smooth,
        )

        self.wait(FAILURE_HOLD)
        self.play(FadeOut(narration), FadeOut(legend), run_time=0.5)

    def play_major_axis(self) -> None:
        """Scene 4: Bianco-style robust resolution on the major (TN) axis."""
        self._ensure_minor_dynamics()
        # Strip the failing minor-axis arrows + error indicator + truth, so we
        # can introduce the major-axis approach without visual collisions.
        self.remove(
            self.resolved_arrow, self.error_indicator,
            self.raw_pos_arrow, self.raw_neg_arrow,
            self.fitter_marker, self.truth_arrow,
        )

        intro = Text(
            "Bianco et al. resolve the 180 deg ambiguity on the MAJOR (TN) axis instead.",
            font_size=22, color=LABEL_COLOR,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(intro), run_time=0.5)

        # Rewind theta to the start of the typical range so the new method
        # gets the same sweep that broke the old one.
        if abs(self.theta_tracker.get_value() - THETA_MAJOR_REWIND) > 0.05:
            self.play(
                self.theta_tracker.animate.set_value(THETA_MAJOR_REWIND),
                run_time=MAJOR_REWIND_TIME, rate_func=smooth,
            )

        # Layer in the major-axis visualization. Note: we do NOT re-add the
        # truth_arrow here -- when the major-axis method works correctly, the
        # derived_gaze_arrow IS truth, so showing both produces collided labels.
        # Scene 3 already established that green = correct gaze.
        self.add(
            self.major_axis_line, self.major_resolved_arrow, self.tn_endpoints,
            self.major_dot_label, self.derived_gaze_arrow,
        )

        explainer = Text(
            "Major axis lies near the body AP -- far from the half-plane boundary.",
            font_size=20, color=LABEL_COLOR,
        ).next_to(intro, UP, buff=0.10)
        self.play(FadeIn(explainer), run_time=0.5)

        # Sweep through (and past) the minor-axis failure boundary -- the
        # major-axis resolution should remain robust the whole way.
        self.play(
            self.theta_tracker.animate.set_value(THETA_MAJOR_END),
            run_time=MAJOR_RUN_TIME, rate_func=smooth,
        )

        verdict = Text(
            "Derived gaze (perpendicular to resolved major) tracks truth past 90 deg.",
            font_size=20, color=TRUTH_COLOR, weight=BOLD,
        ).next_to(explainer, UP, buff=0.10)
        self.play(FadeIn(verdict), run_time=0.5)

        self.wait(MAJOR_HOLD)
        self.play(FadeOut(intro), FadeOut(explainer), FadeOut(verdict), run_time=0.5)

    def play_major_axis_clean(self) -> None:
        """Major-axis resolution introduced from scratch -- for the 'correct way' master.

        Unlike play_major_axis (which strips down minor-axis failure machinery
        from the preceding scenes), this presents the canonical method directly.
        It's the canonical, Bianco-style version of the eye-axis analysis.
        """
        # Replace the static left eye with the rotating fit ellipse. Start
        # the fit ellipse at the resting orientation (theta=90°, gaze fully
        # lateral) so the swap is visually continuous with the static eye.
        if self.eye_left in self.mobjects:
            self.play(FadeOut(self.eye_left), run_time=0.4)

        self.theta_tracker.set_value(np.pi / 2.0)
        self.fitter_sign_tracker.set_value(1.0)

        # Layer in only the canonical-method mobjects -- no raw candidates,
        # no minor-axis flip rule, no truth-vs-resolved comparison, no
        # |m̂ · f̂| margin readout (that's a teaching detail covered in the
        # docs, not needed for the canonical-flow narrative).
        self.add(
            self.fit_ellipse, self.minor_axis_line,
            self.major_axis_line, self.major_resolved_arrow, self.tn_endpoints,
            self.derived_gaze_arrow,
        )

        intro = Text(
            "Each fitted ellipse axis is a directionless line. We resolve the 180° ambiguity on the major (TN) axis.",
            font_size=20, color=LABEL_COLOR,
        ).to_edge(UP, buff=0.4)
        self.play(FadeIn(intro), run_time=0.5)

        explainer = Text(
            "Major axis lies near the body AP — far from the half-plane boundary, so the rule is stable.",
            font_size=18, color=LABEL_COLOR,
        ).next_to(intro, DOWN, buff=0.15)
        self.play(FadeIn(explainer), run_time=0.5)
        self.wait(0.5)

        # First, rotate the eye into a typical converged position. The viewer
        # watches the eye actually move into the orientation that the
        # derived-gaze arrow represents -- so the arrow direction is felt as
        # the consequence of the eye rotation, not just a static label.
        self.play(
            self.theta_tracker.animate.set_value(THETA_TYPICAL_START),
            run_time=1.8, rate_func=smooth,
        )
        self.wait(0.7)

        # Then sweep through the rest of the biological range (rest → past-lateral)
        self.play(
            self.theta_tracker.animate.set_value(THETA_MAJOR_END),
            run_time=MAJOR_RUN_TIME, rate_func=smooth,
        )

        verdict = Text(
            "Gaze is the +90° perpendicular of the resolved major (left eye outward).",
            font_size=20, color=TRUTH_COLOR, weight=BOLD,
        ).next_to(explainer, DOWN, buff=0.15)
        self.play(FadeIn(verdict), run_time=0.5)
        self.wait(MAJOR_HOLD)

        self.play(FadeOut(intro), FadeOut(explainer), FadeOut(verdict), run_time=0.5)

    def play_vergence(self) -> None:
        """Scene 6: Bianco eye-frame vergence demo with both eyes converging symmetrically."""
        # ----- Phase 1: rotate the left eye back to its rest orientation -----
        # If a left-eye fit ellipse is currently on stage (from play_major_axis_*),
        # smoothly rotate it back to theta = 90 deg (the resting lateral gaze).
        # Every always_redraw decoration on top of it (TN axis line, T/N
        # labels, derived gaze, etc.) tracks the rotation, so nothing pops.
        target_theta = np.pi / 2
        if (
            self.fit_ellipse in self.mobjects
            and abs(self.theta_tracker.get_value() - target_theta) > 0.05
        ):
            self.play(
                self.theta_tracker.animate.set_value(target_theta),
                run_time=0.8, rate_func=smooth,
            )

        # ----- Phase 2: fade out only the labels that have no Scene 6 equivalent -----
        # We keep fit_ellipse, major_axis_line, and tn_endpoints visible;
        # they have direct verg_* counterparts and will be invisibly swapped
        # in Phase 3. The eye, TN line, and T/N labels stay continuous.
        fade_redraws = [
            self.minor_axis_line,
            self.major_resolved_arrow,
            self.derived_gaze_arrow,
            self.major_dot_label,
            # Minor-axis failure-mode mobjects (in case we entered via MasterScene)
            self.raw_pos_arrow, self.raw_neg_arrow, self.fitter_marker,
            self.resolved_arrow, self.truth_arrow, self.error_indicator,
        ]
        fade_statics = [
            self.forward_half_shade, self.backward_half_shade,
            self.fwd_label, self.bwd_label, self.halfplane_line,
            self.f_arrow, self.f_label, self.l_arrow, self.l_label,
        ]

        # always_redraw mobjects can't FadeOut cleanly (their updater resets
        # opacity each frame), so snapshot them into static copies first.
        fadeable: list[Mobject] = []
        for m in fade_redraws:
            if m in self.mobjects:
                snap = m.copy()
                snap.clear_updaters()
                self.remove(m)
                self.add(snap)
                fadeable.append(snap)
        for m in fade_statics:
            if m in self.mobjects:
                fadeable.append(m)

        if fadeable:
            self.play(*[FadeOut(m) for m in fadeable], run_time=0.6)

        # ----- Phase 3: invisible swap to the vergence layer -----
        # fit_ellipse, major_axis_line, tn_endpoints get replaced by their
        # convergence-tracker counterparts. At theta=90 deg / convergence=0
        # the orientations match, so the visual is identical. The static
        # eye_right also gets replaced by verg_eye_right (same default pose).
        self.convergence_tracker.set_value(0.0)
        for old in (
            self.fit_ellipse, self.major_axis_line, self.tn_endpoints,
            self.eye_left, self.eye_right,
        ):
            if old in self.mobjects:
                self.remove(old)

        # Cones start invisible (alpha=0) and reveal slowly so they don't pop
        # in. Eyes / TN axes / math readout still appear instantly via add().
        self.verg_cone_alpha.set_value(0.0)
        self.add(
            self.verg_left_cone, self.verg_right_cone, self.verg_binocular,
            self.verg_eye_left, self.verg_eye_right,
            self.verg_left_major, self.verg_right_major,
            self.verg_left_tn, self.verg_right_tn,
            self.verg_readout,
        )
        # Reveal the cones over ~1.5s so the viewer registers each eye's
        # monocular field appearing before the convergence rotation begins.
        self.play(
            self.verg_cone_alpha.animate.set_value(1.0),
            run_time=1.5, rate_func=smooth,
        )

        # Title + epilogue live at the top of the frame; the long tail now
        # occupies the bottom of the frame, so we use the upper area where
        # the cones are pale enough to keep the text readable.
        title = Text(
            "Vergence: total nasal rotation across both eyes",
            font_size=26, color=LABEL_COLOR,
        ).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)
        self.wait(0.5)

        # Animate symmetric nasal convergence from rest (0 deg) to ~30 deg per eye
        self.play(
            self.convergence_tracker.animate.set_value(30.0),
            run_time=6.0, rate_func=smooth,
        )

        epilogue = Text(
            "Equivalent to right_major − left_major in the body frame.",
            font_size=20, color=LABEL_COLOR,
        ).next_to(title, DOWN, buff=0.20)
        self.play(FadeIn(epilogue), run_time=0.5)
        self.wait(2.5)

        self.play(FadeOut(title), FadeOut(epilogue), run_time=0.5)

    def play_boundary(self) -> None:
        """Scene 5: jitter near 90 deg to show the minor-axis flicker."""
        # Strip the major-axis layer; restore the minor-axis-only display.
        self.remove(
            self.major_axis_line, self.major_resolved_arrow, self.tn_endpoints,
            self.major_dot_label, self.derived_gaze_arrow,
        )
        self._ensure_minor_dynamics()
        for m in (self.raw_pos_arrow, self.raw_neg_arrow,
                  self.fitter_marker, self.resolved_arrow, self.truth_arrow):
            if m not in self.mobjects:
                self.add(m)

        # Smoothly shift theta toward the boundary if we're far from it.
        if abs(self.theta_tracker.get_value() - THETA_BOUNDARY_LOW) > 0.05:
            self.play(
                self.theta_tracker.animate.set_value(THETA_BOUNDARY_LOW),
                run_time=1.2, rate_func=smooth,
            )

        narration = Text(
            "Near the 90 deg boundary, fit noise causes minor-axis gaze to flicker by 180 deg.",
            font_size=22, color=LABEL_COLOR,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(narration), run_time=0.5)

        # Rapid oscillation across the half-plane boundary -- each crossing
        # flips the resolved gaze 180 deg.
        for _ in range(5):
            self.play(
                self.theta_tracker.animate.set_value(THETA_BOUNDARY_HIGH),
                run_time=BOUNDARY_FLICKER_HALF, rate_func=linear,
            )
            self.play(
                self.theta_tracker.animate.set_value(THETA_BOUNDARY_LOW),
                run_time=BOUNDARY_FLICKER_HALF, rate_func=linear,
            )

        self.wait(0.8)
        self.play(FadeOut(narration), run_time=0.5)


# ============================================================================
# Per-scene entry points
# ============================================================================

class Scene1Setup(GazeBase):
    def construct(self) -> None:
        self.build_objects()
        self.play_definitions()
        self.play_setup(animated=True)


class Scene2Typical(GazeBase):
    def construct(self) -> None:
        self.build_objects()
        self.play_setup(animated=False)
        self.play_typical()


class Scene3Failure(GazeBase):
    def construct(self) -> None:
        self.build_objects()
        self.play_setup(animated=False)
        # Begin at the end of the typical range so we can rotate past 90 deg.
        self.theta_tracker.set_value(THETA_TYPICAL_END)
        self.play_failure()


class Scene4MajorAxis(GazeBase):
    def construct(self) -> None:
        self.build_objects()
        self.play_setup(animated=False)
        # Begin where the failure scene ended, so the major-axis "redo" is
        # visually the same starting state.
        self.theta_tracker.set_value(THETA_FAILURE_END)
        self.play_major_axis()


class Scene5Boundary(GazeBase):
    def construct(self) -> None:
        self.build_objects()
        self.play_setup(animated=False)
        self.theta_tracker.set_value(THETA_BOUNDARY_LOW)
        self.play_boundary()


class Scene6Vergence(GazeBase):
    def construct(self) -> None:
        self.build_objects()
        self.play_setup(animated=False)
        self.play_vergence()


class MasterCorrectScene(GazeBase):
    """The 'correct way' sequence: definitions, setup, major-axis resolution, vergence.

    Skips the minor-axis failure-mode demonstrations (Scenes 2/3/5) and shows
    only the canonical Bianco-style approach end-to-end. Use this for talks
    where you want to teach the right method without showing the bug.
    """

    def construct(self) -> None:
        self.build_objects()
        self.play_definitions()
        self.play_setup(animated=True)
        self.play_major_axis_clean()
        self.play_vergence()
        self.wait(1.0)


class MasterScene(GazeBase):
    """The full sequence including failure-mode scenes -- pedagogical for the
    'why we needed to refactor v5' story.
    """

    def construct(self) -> None:
        self.build_objects()
        self.play_definitions()
        self.play_setup(animated=True)
        self.play_typical()
        self.play_failure()
        self.play_major_axis()
        self.play_boundary()
        self.play_vergence()
        self.wait(1.0)
