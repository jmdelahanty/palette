"""Manim scenes for the track-kinematics speed pipeline.

Family 2 in ``apps/manim/ANIMATIONS_PLAN.md``. Each scene targets one
transformation in ``compute_speed.py``:

  Scene21FramePathDistance  : raw per-frame Euclidean step length

Future scenes (placeholders, not yet built):
  Scene22Hysteresis         : hysteresis state machine + filtered output
  Scene23Smoothing          : moving-average / Savitzky-Golay smoothing
  Scene24CausalExponential  : causal exponential kernel for segmentation
  Scene25Acceleration       : speed → acceleration → smoothed acceleration

Render in the ``manim`` env; cache must be built first via the
``palette`` env (see ``apps/manim/data/build_cache.py``).
"""

from __future__ import annotations

import numpy as np
from manim import (
    Arrow,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    ManimColor,
    NumberPlane,
    RIGHT,
    Text,
    UP,
    VGroup,
    Write,
    config,
)

from apps.manim._common.timeseries import TimeseriesScene


# Mini-arena view geometry (Manim scene units).
ARENA_WIDTH_UNITS = 6.5
ARENA_HEIGHT_UNITS = 4.0
ARENA_CENTER_OFFSET = np.array([-2.5, -0.4, 0.0])

DOT_COLOR_PREV = "#a3c8ff"
DOT_COLOR_NEXT = "#1f6feb"
ARROW_COLOR = "#27ae60"
TELEPORT_COLOR = "#e74c3c"


class Scene21FramePathDistance(TimeseriesScene):
    """Frame path distance (raw) — the first thing the speed pipeline computes.

    Beats:
      1. Title + one-line gloss.
      2. Mini-arena view of a slice of the highest-peak bout's trajectory;
         positions appear as dots, ordered.
      3. Zoom to a single pair (p_i, p_{i+1}); draw displacement vector.
      4. Annotate ``|p_{i+1} - p_i|`` and the computed pixel length.
      5. Step the arrow forward through three consecutive frames inside the
         bout; the length value updates each step.
      6. Demonstrate the >500 px teleport guard with a synthetic teleport
         pair; the displacement vector turns red and a "rejected" badge
         appears.
    """

    def construct(self) -> None:
        title = Text("Frame path distance", font_size=42).to_edge(UP, buff=0.45)
        gloss = Text(
            "Per-frame Euclidean step length |pᵢ₊₁ − pᵢ| "
            "between consecutive detections.",
            font_size=22,
        ).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), run_time=0.7)
        self.play(FadeIn(gloss), run_time=0.5)
        self.wait(0.4)

        # --- Beat 2: arena view.
        # Find the highest-peak bout in the cache (works for synthetic and
        # real caches alike); use frames around the peak for the focus.
        bout_start_s, bout_end_s, peak_frame_idx = self.find_peak_bout(level="filtered")
        t = self.cache.t
        # Pick a small window around the peak for trail dots.
        bout_mask = (t >= bout_start_s) & (t <= bout_end_s)
        bout_idx_all = np.where(bout_mask)[0]
        # Trail = up to 14 evenly-spaced samples within the bout.
        step = max(1, len(bout_idx_all) // 14)
        trail_idx = bout_idx_all[::step][:14]
        trail_positions = self.cache.position_xy[trail_idx]

        # Focus 5 truly-consecutive frames starting just before the peak so
        # the arrow steps trace through the rising edge.
        focus_start = max(0, peak_frame_idx - 2)
        if focus_start + 5 > len(t):
            focus_start = len(t) - 5
        consecutive_frames = [focus_start + k for k in range(5)]
        consecutive_positions = self.cache.position_xy[consecutive_frames]

        # Combine trail + consecutive into a single set for the bbox so
        # everything fits in the arena view.
        all_positions_for_bbox = np.concatenate(
            [trail_positions, consecutive_positions], axis=0
        )

        plane, trail_dots, _ = self._build_arena_view(
            all_positions_for_bbox, trail_idx, only_render_first_n=len(trail_positions)
        )
        consec_dots = VGroup()
        for px_xy in consecutive_positions:
            consec_dots.add(
                Dot(
                    point=plane._px_to_plane(px_xy),  # type: ignore[attr-defined]
                    radius=0.075,
                    color=ManimColor(DOT_COLOR_NEXT),
                )
            )

        arena_caption = Text(
            "fish position during the highest-peak bout in the window "
            "(trail = sampled, highlighted = 5 consecutive frames)",
            font_size=18,
        ).next_to(plane, DOWN, buff=0.20)

        self.play(FadeIn(plane), run_time=0.5)
        self.play(FadeIn(arena_caption), run_time=0.4)
        for dot in trail_dots:
            self.add(dot)
            self.wait(0.05)
        self.wait(0.2)
        self.play(FadeIn(consec_dots), run_time=0.4)
        self.wait(0.3)

        # --- Beat 3-4: arrow between two truly consecutive frames; value from cache.
        frame_a = consecutive_frames[0]
        frame_b = consecutive_frames[1]
        pos_a_px = self.cache.position_xy[frame_a]
        pos_b_px = self.cache.position_xy[frame_b]
        cached_distance_px = float(self.cache.frame_path_distance_raw_px[frame_b])

        arrow = self._make_displacement_arrow(plane, pos_a_px, pos_b_px)
        length_label = self._make_length_label(arrow, cached_distance_px)
        frame_label = self._make_frame_pair_label(plane, frame_a, frame_b)

        self.play(FadeOut(gloss), run_time=0.3)
        self.play(FadeIn(frame_label), run_time=0.4)
        self.play(FadeIn(arrow), run_time=0.5)
        self.play(Write(length_label), run_time=0.6)
        self.wait(0.7)

        # --- Beat 5: step through 3 more consecutive pairs (frame+1..2, +2..3, +3..4).
        for step in range(1, 4):
            new_a = consecutive_frames[step]
            new_b = consecutive_frames[step + 1]
            new_pos_a = self.cache.position_xy[new_a]
            new_pos_b = self.cache.position_xy[new_b]
            new_distance = float(self.cache.frame_path_distance_raw_px[new_b])
            new_arrow = self._make_displacement_arrow(plane, new_pos_a, new_pos_b)
            new_length = self._make_length_label(new_arrow, new_distance)
            new_frame_label = self._make_frame_pair_label(plane, new_a, new_b)
            self.play(
                FadeOut(arrow, run_time=0.2),
                FadeOut(length_label, run_time=0.2),
                FadeOut(frame_label, run_time=0.2),
            )
            self.play(
                FadeIn(new_arrow, run_time=0.25),
                FadeIn(new_length, run_time=0.25),
                FadeIn(new_frame_label, run_time=0.25),
            )
            arrow, length_label, frame_label = new_arrow, new_length, new_frame_label
            self.wait(0.30)

        self.wait(0.4)

        # --- Beat 6: teleport guard (still in arena view).
        self.play(
            FadeOut(arrow),
            FadeOut(length_label),
            FadeOut(frame_label),
            run_time=0.4,
        )

        guard_caption = Text(
            "Validity guard: any displacement > 500 px is rejected (treated as 0).",
            font_size=22,
        ).next_to(arena_caption, DOWN, buff=0.22)
        self.play(FadeIn(guard_caption), run_time=0.4)

        teleport_pos_a = self.cache.position_xy[frame_a]
        teleport_pos_b = teleport_pos_a + np.array([600.0, 0.0])
        teleport_arrow = self._make_displacement_arrow(
            plane, teleport_pos_a, teleport_pos_b, color=TELEPORT_COLOR, allow_offscreen=True
        )
        teleport_label = self._make_length_label(
            teleport_arrow, 600.0, color=TELEPORT_COLOR, suffix=" → rejected"
        )
        self.play(FadeIn(teleport_arrow), run_time=0.4)
        self.play(Write(teleport_label), run_time=0.5)
        self.wait(1.0)

        # --- Beat 7: speed graph payoff.
        # Fade the arena and bring up the speed_raw curve over the clean bout.
        self.play(
            FadeOut(
                VGroup(
                    plane,
                    trail_dots,
                    consec_dots,
                    arena_caption,
                    guard_caption,
                    teleport_arrow,
                    teleport_label,
                )
            ),
            run_time=0.6,
        )

        self._play_speed_graph_beat(frame_a, consecutive_frames)

        self.play(FadeOut(title), run_time=0.4)

    def _play_speed_graph_beat(
        self, frame_a: int, consecutive_frames: list[int]
    ) -> None:
        """Closing beat: cached speed_raw curve over the focus bout, with
        the 4 stepped frames highlighted on the curve."""
        from manim import Create

        bout_start_s, bout_end_s, _ = self.find_peak_bout(level="filtered")
        # Display window: a bit before/after the bout for context.
        x_min = max(float(self.cache.t[0]), bout_start_s - 0.20)
        x_max = min(self.cache.duration_s, bout_end_s + 0.30)
        # Pick a y-axis ceiling that comfortably exceeds the local peak.
        local_max = float(np.nanmax(self.cache.speed_raw_mm_s[
            (self.cache.t >= x_min) & (self.cache.t <= x_max)
        ]))
        y_ceiling = float(np.ceil((local_max * 1.15) / 30.0) * 30.0)
        x_step = round(max(0.05, (x_max - x_min) / 5), 2)

        axes, axes_group = self.make_axes(
            x_range=(x_min, x_max, x_step),
            y_range=(0.0, y_ceiling, 30.0),
            x_length=10.0,
            y_length=3.4,
            x_label="time (s)",
            y_label="speed_raw (mm/s)",
        )
        axes_group.shift(np.array([0.0, -0.6, 0.0]))
        self.play(FadeIn(axes_group), run_time=0.6)

        # Slice cache to the display window and plot the cached speed_raw.
        t = self.cache.t
        in_window = (t >= x_min) & (t <= x_max)
        polylines = self.plot_trace(
            axes,
            t=t[in_window],
            y=self.cache.speed_raw_mm_s[in_window],
            color="#1f6feb",
            stroke_width=2.6,
        )
        self.play(Create(polylines), run_time=1.6)
        self.wait(0.3)

        # Mark the 4 stepped frames on the curve.
        markers = VGroup()
        last_label: Text | None = None
        for k, b in enumerate(consecutive_frames[1:], start=1):
            t_b = float(self.cache.t[b])
            speed_b = float(self.cache.speed_raw_mm_s[b])
            point = axes.coords_to_point(t_b, speed_b)
            dot = Dot(point=point, radius=0.07, color=ManimColor("#27ae60"))
            markers.add(dot)
            self.play(FadeIn(dot), run_time=0.18)
            # Last marker also shows a value label.
            if k == len(consecutive_frames) - 1:
                last_label = Text(
                    f"speed_raw[{b}] = {speed_b:0.1f} mm/s",
                    font_size=22,
                    color=ManimColor("#27ae60"),
                ).next_to(dot, UP, buff=0.18)
                self.play(Write(last_label), run_time=0.5)

        # Final math line under the axes.
        delta_s = float(self.cache.delta_seconds[consecutive_frames[-1]])
        distance_px = float(
            self.cache.frame_path_distance_raw_px[consecutive_frames[-1]]
        )
        math_line = Text(
            f"= {distance_px:0.2f} px ÷ {delta_s * 1000:0.2f} ms ÷ "
            f"{self.cache.px_per_mm:0.0f} px·mm⁻¹",
            font_size=22,
            color=ManimColor("#cccccc"),
        ).next_to(axes_group, DOWN, buff=0.22)
        self.play(FadeIn(math_line), run_time=0.5)
        self.wait(1.6)

        cleanup = VGroup(axes_group, polylines, markers, math_line)
        if last_label is not None:
            cleanup.add(last_label)
        self.play(FadeOut(cleanup), run_time=0.5)

    # ---------------------------------------------------------------- helpers

    def _build_arena_view(
        self,
        positions_px: np.ndarray,
        frame_indices: np.ndarray,
        *,
        only_render_first_n: int | None = None,
    ) -> tuple[NumberPlane, VGroup, dict[int, Dot]]:
        """Return a NumberPlane plus dots for each frame's position.

        ``positions_px`` may include extra positions used only to size the
        bbox; pass ``only_render_first_n`` to limit which positions get
        materialized as dot mobjects in the returned VGroup.

        Maps the pixel positions into a small Manim plane so the trajectory
        is visible at a comfortable scale.
        """
        # Compute a px-bbox with margin so all dots fit; map to plane units.
        x_min, y_min = positions_px.min(axis=0) - 5.0
        x_max, y_max = positions_px.max(axis=0) + 5.0
        bbox_w = max(x_max - x_min, 1e-3)
        bbox_h = max(y_max - y_min, 1e-3)

        # Fit-to-arena scale so the longer side of the bbox spans the arena.
        scale = min(ARENA_WIDTH_UNITS / bbox_w, ARENA_HEIGHT_UNITS / bbox_h)

        def px_to_plane(px_xy: np.ndarray) -> np.ndarray:
            local = (px_xy - np.array([x_min, y_min])) * scale
            local_3d = np.array([local[0], local[1], 0.0])
            # Center the bbox inside the arena rectangle.
            arena_offset = np.array(
                [
                    (ARENA_WIDTH_UNITS - bbox_w * scale) / 2.0,
                    (ARENA_HEIGHT_UNITS - bbox_h * scale) / 2.0,
                    0.0,
                ]
            )
            return local_3d + arena_offset + ARENA_CENTER_OFFSET

        plane = NumberPlane(
            x_range=[0, ARENA_WIDTH_UNITS, 1],
            y_range=[0, ARENA_HEIGHT_UNITS, 1],
            x_length=ARENA_WIDTH_UNITS,
            y_length=ARENA_HEIGHT_UNITS,
            background_line_style={
                "stroke_color": "#3a3a3a",
                "stroke_opacity": 0.35,
                "stroke_width": 1.0,
            },
            axis_config={
                "stroke_color": "#5a5a5a",
                "stroke_width": 1.0,
                "include_numbers": False,
            },
        ).move_to(
            ARENA_CENTER_OFFSET
            + np.array([ARENA_WIDTH_UNITS / 2.0, ARENA_HEIGHT_UNITS / 2.0, 0.0])
        )

        # Stash the converter so other helpers can reuse it.
        plane._px_to_plane = px_to_plane  # type: ignore[attr-defined]

        dots = VGroup()
        idx_to_dot: dict[int, Dot] = {}
        n_render = only_render_first_n if only_render_first_n is not None else len(positions_px)
        for k, (px_xy, frame_index) in enumerate(zip(positions_px, frame_indices)):
            if k >= n_render:
                break
            point = px_to_plane(px_xy)
            dot = Dot(point=point, radius=0.06, color=ManimColor(DOT_COLOR_PREV))
            dots.add(dot)
            idx_to_dot[int(frame_index)] = dot
        return plane, dots, idx_to_dot

    def _make_displacement_arrow(
        self,
        plane: NumberPlane,
        pos_a_px: np.ndarray,
        pos_b_px: np.ndarray,
        *,
        color: str = ARROW_COLOR,
        allow_offscreen: bool = False,
    ) -> Arrow:
        a_point = plane._px_to_plane(pos_a_px)  # type: ignore[attr-defined]
        b_point = plane._px_to_plane(pos_b_px)  # type: ignore[attr-defined]
        if allow_offscreen:
            # Clip the tip back inside the arena so the arrow stays visible.
            arena_max = ARENA_CENTER_OFFSET + np.array(
                [ARENA_WIDTH_UNITS, ARENA_HEIGHT_UNITS, 0.0]
            )
            arena_min = ARENA_CENTER_OFFSET
            b_point = np.minimum(np.maximum(b_point, arena_min + 0.05), arena_max - 0.05)
        return Arrow(
            start=a_point,
            end=b_point,
            buff=0.0,
            stroke_width=4.0,
            color=ManimColor(color),
            max_tip_length_to_length_ratio=0.18,
        )

    def _make_length_label(
        self,
        arrow: Arrow,
        displacement_px: float,
        *,
        color: str = ARROW_COLOR,
        suffix: str = "",
    ) -> Text:
        text = Text(
            f"|pᵢ₊₁ − pᵢ| = {displacement_px:0.2f} px{suffix}",
            font_size=24,
            color=ManimColor(color),
        )
        midpoint = (arrow.get_start() + arrow.get_end()) / 2.0
        text.next_to(midpoint, UP, buff=0.18)
        return text

    def _make_frame_pair_label(
        self, plane: NumberPlane, frame_a: int, frame_b: int
    ) -> Text:
        return Text(
            f"frame {frame_a} → frame {frame_b}",
            font_size=20,
            color=ManimColor("#cccccc"),
        ).next_to(plane, UP, buff=0.18)
