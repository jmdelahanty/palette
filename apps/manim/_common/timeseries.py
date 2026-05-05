"""Shared base scene for time-series teaching animations.

Used by the speed-pipeline and bout-detection family scene files. Loads
the pre-computed ``.npz`` cache (built via ``apps/manim/data/build_cache``)
so the manim render env never needs ``fisheye``/``zarr``.

Conventions:
- X axis is **seconds**. A "frame N" tick can be requested for single-frame
  events (hysteresis debounce, threshold crossings) where the discrete
  timeline matters.
- Y axis is **mm/s** for speed (cache stores px/s; we divide by
  ``px_per_mm`` here).
- Regions from the synthetic trace get a fixed color palette so the same
  region keeps the same hue across scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from manim import (
    Axes,
    Dot,
    DOWN,
    LEFT,
    Line,
    ManimColor,
    RIGHT,
    Rectangle,
    Scene,
    Text,
    UP,
    VGroup,
    WHITE,
)

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
DEFAULT_CACHE_NAME = "real_swim_bout"
DEFAULT_CACHE_PATH = CACHE_DIR / f"{DEFAULT_CACHE_NAME}.npz"


def cache_path_for(name: str) -> Path:
    return CACHE_DIR / f"{name}.npz"


REGION_PALETTE: Mapping[str, str] = {
    "still_pre": "#3a3a3a",
    "clean_bout": "#27ae60",
    "noisy_threshold": "#e8a33d",
    "multi_peak_bout": "#1f6feb",
    "gap": "#7a3a8a",
    "still_post": "#3a3a3a",
}


@dataclass
class Cache:
    """In-memory view of the .npz cache, with mm/s conversions precomputed."""

    fps: float
    px_per_mm: float
    frames: np.ndarray
    t: np.ndarray
    position_xy: np.ndarray
    heading_deg: np.ndarray
    delta_frames: np.ndarray
    delta_seconds: np.ndarray
    transition_valid: np.ndarray
    transition_reason_code: np.ndarray
    speed_raw_mm_s: np.ndarray
    speed_filtered_mm_s: np.ndarray
    speed_smoothed_mm_s: np.ndarray
    speed_averaged_mm_s: np.ndarray
    frame_path_distance_raw_px: np.ndarray
    frame_path_distance_filtered_px: np.ndarray
    frame_path_distance_smoothed_px: np.ndarray
    cumulative_path_distance_px: np.ndarray
    region_labels: list[str]
    region_starts_s: np.ndarray
    region_ends_s: np.ndarray
    ground_truth_bouts: np.ndarray  # (n, 2) start_s, end_s
    smooth_seconds: float
    distance_smooth_seconds: float
    hysteresis_high_px: float
    hysteresis_low_px: float
    hysteresis_min_frames: int

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    def region_window(self, label: str) -> tuple[float, float]:
        idx = self.region_labels.index(label)
        return float(self.region_starts_s[idx]), float(self.region_ends_s[idx])

    def speed(self, level: str) -> np.ndarray:
        """Look up a speed level by name (raw / filtered / smoothed / averaged)."""
        return getattr(self, f"speed_{level}_mm_s")


def load_default_cache(path: Path = DEFAULT_CACHE_PATH) -> Cache:
    if not path.exists():
        raise FileNotFoundError(
            f"Cache missing at {path}. Build it via "
            f"`PYTHONPATH=src ~/miniconda3/envs/palette/bin/python "
            f"-m apps.manim.data.build_cache`."
        )
    raw = np.load(path, allow_pickle=True)
    px_per_mm = float(raw["px_per_mm"])
    return Cache(
        fps=float(raw["fps"]),
        px_per_mm=px_per_mm,
        frames=np.asarray(raw["frames"]),
        t=np.asarray(raw["t"], dtype=np.float64),
        position_xy=np.asarray(raw["position_xy"], dtype=np.float64),
        heading_deg=np.asarray(raw["heading_deg"], dtype=np.float64),
        delta_frames=np.asarray(raw["delta_frames"]),
        delta_seconds=np.asarray(raw["delta_seconds"], dtype=np.float64),
        transition_valid=np.asarray(raw["transition_valid"], dtype=bool),
        transition_reason_code=np.asarray(raw["transition_reason_code"]),
        speed_raw_mm_s=np.asarray(raw["speed_raw"], dtype=np.float64) / px_per_mm,
        speed_filtered_mm_s=np.asarray(raw["speed_filtered"], dtype=np.float64) / px_per_mm,
        speed_smoothed_mm_s=np.asarray(raw["speed_smoothed"], dtype=np.float64) / px_per_mm,
        speed_averaged_mm_s=np.asarray(raw["speed_averaged"], dtype=np.float64) / px_per_mm,
        frame_path_distance_raw_px=np.asarray(raw["frame_path_distance_raw"], dtype=np.float64),
        frame_path_distance_filtered_px=np.asarray(raw["frame_path_distance_filtered"], dtype=np.float64),
        frame_path_distance_smoothed_px=np.asarray(raw["frame_path_distance_smoothed"], dtype=np.float64),
        cumulative_path_distance_px=np.asarray(raw["cumulative_path_distance"], dtype=np.float64),
        region_labels=[str(s) for s in raw["region_labels"]],
        region_starts_s=np.asarray(raw["region_starts_s"], dtype=np.float64),
        region_ends_s=np.asarray(raw["region_ends_s"], dtype=np.float64),
        ground_truth_bouts=np.asarray(raw["ground_truth_bouts"], dtype=np.float64),
        smooth_seconds=float(raw["smooth_seconds"]),
        distance_smooth_seconds=float(raw["distance_smooth_seconds"]),
        hysteresis_high_px=float(raw["hysteresis_high_px"]),
        hysteresis_low_px=float(raw["hysteresis_low_px"]),
        hysteresis_min_frames=int(raw["hysteresis_min_frames"]),
    )


class TimeseriesScene(Scene):
    """Base scene with cache + axis/trace helpers.

    Subclasses use ``self.cache`` for data and ``self.make_axes()`` /
    ``self.plot_trace()`` / ``self.shade_region()`` to build visuals. The
    scene class itself does not call ``construct`` — subclasses must.
    """

    #: Cache file (without extension) to load. Subclasses can override.
    CACHE_NAME: str = DEFAULT_CACHE_NAME

    def setup(self) -> None:
        self.cache = load_default_cache(cache_path_for(self.CACHE_NAME))

    def find_peak_bout(self, *, level: str = "filtered") -> tuple[float, float, int]:
        """Return (start_s, end_s, peak_frame_index) for the highest-peak
        bout in ``cache.ground_truth_bouts`` using the selected speed level.
        """
        speeds = self.cache.speed(level)
        bouts = self.cache.ground_truth_bouts
        if bouts.size == 0:
            raise RuntimeError("No ground_truth_bouts in cache")
        best = None
        best_peak = -np.inf
        for s, e in bouts:
            mask = (self.cache.t >= s) & (self.cache.t <= e)
            if not np.any(mask):
                continue
            local_max = float(np.nanmax(speeds[mask]))
            if local_max > best_peak:
                best_peak = local_max
                peak_idx = int(np.argmax(np.where(mask, speeds, -np.inf)))
                best = (float(s), float(e), peak_idx)
        if best is None:
            raise RuntimeError("All bout windows empty")
        return best

    # --- Axes -----------------------------------------------------------------
    def make_axes(
        self,
        *,
        x_range: tuple[float, float, float] = (0.0, 8.0, 1.0),
        y_range: tuple[float, float, float] = (0.0, 180.0, 30.0),
        x_length: float = 11.0,
        y_length: float = 4.5,
        x_label: str = "time (s)",
        y_label: str = "speed (mm/s)",
        x_tick_format: str = "{:0.1f}",
        y_tick_format: str = "{:0.0f}",
    ) -> tuple[Axes, VGroup]:
        # include_numbers=False because the manim env has no LaTeX; we add
        # plain Text tick labels below.
        axes = Axes(
            x_range=list(x_range),
            y_range=list(y_range),
            x_length=x_length,
            y_length=y_length,
            tips=False,
            axis_config={
                "stroke_color": WHITE,
                "stroke_width": 2.0,
                "include_numbers": False,
            },
        )

        x_ticks = self._add_tick_labels(
            axes, axis="x", value_range=x_range, fmt=x_tick_format
        )
        y_ticks = self._add_tick_labels(
            axes, axis="y", value_range=y_range, fmt=y_tick_format
        )

        x_caption = Text(x_label, font_size=22).next_to(axes.x_axis, DOWN, buff=0.55)
        y_caption = (
            Text(y_label, font_size=22)
            .rotate(np.pi / 2)
            .next_to(axes.y_axis, LEFT, buff=0.55)
        )
        return axes, VGroup(axes, x_ticks, y_ticks, x_caption, y_caption)

    @staticmethod
    def _add_tick_labels(
        axes: Axes,
        *,
        axis: str,
        value_range: tuple[float, float, float],
        fmt: str,
    ) -> VGroup:
        start, end, step = value_range
        labels = VGroup()
        # Inclusive of end if the step lands on it.
        n_steps = int(round((end - start) / step))
        for i in range(n_steps + 1):
            value = start + i * step
            if axis == "x":
                point = axes.coords_to_point(value, value_range[0] if False else axes.y_range[0])
                anchor = DOWN
            else:
                point = axes.coords_to_point(axes.x_range[0], value)
                anchor = LEFT
            text = Text(fmt.format(value), font_size=18)
            text.next_to(point, anchor, buff=0.10)
            labels.add(text)
        return labels

    # --- Region shading -------------------------------------------------------
    def shade_region(
        self,
        axes: Axes,
        label: str,
        *,
        color: str | None = None,
        opacity: float = 0.12,
    ) -> Rectangle:
        start_s, end_s = self.cache.region_window(label)
        return self._shade_interval(
            axes, start_s, end_s, color=color or REGION_PALETTE[label], opacity=opacity
        )

    def shade_bout(
        self,
        axes: Axes,
        start_s: float,
        end_s: float,
        *,
        color: str = "#27ae60",
        opacity: float = 0.18,
    ) -> Rectangle:
        return self._shade_interval(axes, start_s, end_s, color=color, opacity=opacity)

    def _shade_interval(
        self,
        axes: Axes,
        start_s: float,
        end_s: float,
        *,
        color: str,
        opacity: float,
    ) -> Rectangle:
        y_min = axes.y_range[0]
        y_max = axes.y_range[1]
        bottom_left = axes.coords_to_point(start_s, y_min)
        top_right = axes.coords_to_point(end_s, y_max)
        width = top_right[0] - bottom_left[0]
        height = top_right[1] - bottom_left[1]
        rect = Rectangle(
            width=width,
            height=height,
            stroke_width=0,
            fill_color=ManimColor(color),
            fill_opacity=opacity,
        )
        rect.move_to((bottom_left + top_right) / 2.0)
        rect.set_z_index(-5)
        return rect

    # --- Traces ---------------------------------------------------------------
    def plot_trace(
        self,
        axes: Axes,
        *,
        t: np.ndarray | None = None,
        y: np.ndarray,
        color: str = "#1f6feb",
        stroke_width: float = 2.6,
        mask: np.ndarray | None = None,
    ) -> VGroup:
        """Plot a polyline trace over ``axes``.

        ``mask`` (boolean) selects which samples to include; gaps in the
        mask split the polyline into separate segments so that pipeline
        transitions flagged invalid don't produce phantom connecting lines.
        """
        if t is None:
            t = self.cache.t
        if mask is None:
            mask = np.isfinite(y)
        else:
            mask = mask & np.isfinite(y)

        segments = _split_into_segments(t, y, mask)
        polylines = VGroup()
        for seg_t, seg_y in segments:
            if len(seg_t) < 2:
                continue
            points = [axes.coords_to_point(float(ti), float(yi)) for ti, yi in zip(seg_t, seg_y)]
            polyline = VGroup()
            for a, b in zip(points[:-1], points[1:]):
                polyline.add(Line(a, b, stroke_color=ManimColor(color), stroke_width=stroke_width))
            polylines.add(polyline)
        return polylines

    # --- Markers --------------------------------------------------------------
    def frame_marker(
        self,
        axes: Axes,
        frame_index: int,
        y_value: float,
        *,
        color: str = "#e74c3c",
        radius: float = 0.07,
        with_label: bool = True,
        label_above: bool = True,
    ) -> VGroup:
        """Dot at (t_of_frame, y_value) with optional 'frame N' label."""
        t_of_frame = float(frame_index) / self.cache.fps
        point = axes.coords_to_point(t_of_frame, y_value)
        dot = Dot(point=point, radius=radius, color=ManimColor(color))
        if not with_label:
            return VGroup(dot)
        label = Text(f"frame {frame_index}", font_size=18, color=ManimColor(color))
        label.next_to(dot, UP if label_above else DOWN, buff=0.10)
        return VGroup(dot, label)

    def threshold_line(
        self,
        axes: Axes,
        y_value_mm_s: float,
        *,
        color: str = "#e74c3c",
        stroke_width: float = 1.6,
        dashed: bool = True,
        label: str | None = None,
    ) -> VGroup:
        x_min, x_max = axes.x_range[0], axes.x_range[1]
        a = axes.coords_to_point(x_min, y_value_mm_s)
        b = axes.coords_to_point(x_max, y_value_mm_s)
        line = Line(a, b, stroke_color=ManimColor(color), stroke_width=stroke_width)
        if dashed:
            from manim import DashedVMobject

            line = DashedVMobject(line, num_dashes=40)
        group = VGroup(line)
        if label is not None:
            label_text = Text(label, font_size=18, color=ManimColor(color))
            label_text.next_to(b, RIGHT, buff=0.10)
            group.add(label_text)
        return group

    # --- Convenience ----------------------------------------------------------
    def add_region_legend(
        self,
        labels: Iterable[str] = ("clean_bout", "noisy_threshold", "multi_peak_bout"),
    ) -> VGroup:
        rows = VGroup()
        for label in labels:
            swatch = Rectangle(
                width=0.35,
                height=0.22,
                stroke_width=0,
                fill_color=ManimColor(REGION_PALETTE[label]),
                fill_opacity=0.5,
            )
            text = Text(label.replace("_", " "), font_size=18)
            row = VGroup(swatch, text).arrange(RIGHT, buff=0.18)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        rows.to_corner(UP + LEFT, buff=0.30)
        return rows


def _split_into_segments(
    t: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split (t, y) into contiguous runs where ``mask`` is True."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    in_run = False
    start = 0
    for i, ok in enumerate(mask):
        if ok and not in_run:
            in_run = True
            start = i
        elif not ok and in_run:
            in_run = False
            if i - start >= 2:
                segments.append((t[start:i], y[start:i]))
    if in_run and len(mask) - start >= 2:
        segments.append((t[start:], y[start:]))
    return segments
