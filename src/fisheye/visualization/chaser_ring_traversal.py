"""Animate the fish traversing the responsive rings around an object, one entry at a time,
with every bout of the entry drawn as its own segment.

This ties three separate findings into one picture:

  * RINGS. The object is ringed by concentric distance bands (the `chaser_bout_response`
    distance bins). The **responsive shell, 8-16 mm**, is shaded: that is where the steering
    dose-response peaks (+0.574 at 10 mm, +0.551 at 14 mm) and where escapes fire (median
    onset 13.5 mm). Outside it, nothing much happens. The rings are not decoration -- they
    are the axis the whole analysis is measured on.

  * ENTRIES. The fish visits the near zone a handful of times per epoch (a "visit" =
    entering 15 mm and not leaving until it passes 20 mm). Each entry is animated separately,
    because a median over 4-10 entries is noise -- the honest object is the trajectory.

  * BOUTS PER ENTRY. Swimming is discrete: the fish moves in bouts, then coasts. Each bout in
    an entry is drawn as its own segment and lights up as it is executed, so "a set of bouts
    per entry" is literal. **Escape bouts (peak > 100 mm/s) are red and thick**; ordinary
    bouts are blue. The set of bouts is what the fish actually *did* in the shell.

FRAME (inherited from chaser_visit_trajectories)
------------------------------------------------
Object at the origin, arena centre rotated onto +x, so pre and post entries -- whose objects
sit at different arena positions -- land on top of each other and are comparable. The wall is
the grey arc it really is; since the object sits ~7 mm from it, a wall-following fish sweeps
through the rings for free, and that confound is drawn rather than hidden.

Static epochs only (pre/post). During the chase the object moves, so a ring centred on its
median position would be a fiction; the chase-epoch escape dynamics are a chaser-centric
question and live in `chaser_escape_events`.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402

from fisheye.analysis.chaser_bout_response import _segment_visits
from fisheye.analysis.chaser_escape_events import DEFAULT_PEAK_SPEED_THRESHOLD_MM_S
from fisheye.analysis.chaser_escape_freeze import (
    _heading_angles_from_chaser,
    chaser_frame_transform,
)
from fisheye.analysis.chaser_radial_occupancy import (
    _apply_settle_trim,
    _decode_text_column,
    _open_root,
    _protocol_position_transition_s,
    _read_epochs,
    _resolve_chaser_distance_run,
    _safe_float,
)
from fisheye.visualization.chaser_visit_trajectories import (
    DEFAULT_EPOCHS,
    DEFAULT_VIRTUAL_ROTATIONS_DEG,
    DOT_RADIUS_MM,
    VisitScene,
    collect_visits,
)


RING_EDGES_MM = (4.0, 8.0, 12.0, 16.0, 20.0, 25.0)
RESPONSIVE_BAND_MM = (8.0, 16.0)       # steering dose-response peak + escape trigger live here
ESCAPE_TRIGGER_MM = 13.5               # median escape onset distance, cohort
ESCAPE_C = "#dc2626"
BOUT_C = "#2563eb"
RESP_C = "#f59e0b"


def collect_ring_entries(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    swim_bout_run: str = "latest",
    bout_response_component: str = "latest",
    epochs_wanted: Sequence[str] = DEFAULT_EPOCHS,
    virtual_rotations_deg: Sequence[float] = DEFAULT_VIRTUAL_ROTATIONS_DEG,
    peak_speed_threshold_mm_s: float = DEFAULT_PEAK_SPEED_THRESHOLD_MM_S,
    **visit_kwargs: Any,
) -> tuple[list[VisitScene], dict[str, Any]]:
    """collect_visits, then attach the bout SEGMENTS of each entry (with peak speed).

    The base collector marks bout *onsets*; here each visit also gets a ``bouts`` list, one
    entry per bout that starts inside the visit window:

        {"xy": (m, 2) canonical-frame segment, "peak_speed_mm_s", "is_escape",
         "i0", "i1"}  # i0/i1 index into visit["xy"]

    so the animation can light bouts up one at a time and colour escapes.
    """

    scenes, meta = collect_visits(
        zarr_path,
        chaser_distance_run=chaser_distance_run,
        swim_bout_run=swim_bout_run,
        epochs_wanted=epochs_wanted,
        virtual_rotations_deg=virtual_rotations_deg,
        **visit_kwargs,
    )

    bs, be, peak = _load_bout_segments(zarr_path, chaser_distance_run, bout_response_component)
    meta["peak_speed_threshold_mm_s"] = float(peak_speed_threshold_mm_s)
    meta["responsive_band_mm"] = list(RESPONSIVE_BAND_MM)
    meta["ring_edges_mm"] = list(RING_EDGES_MM)

    for scene in scenes:
        for visit in scene.visits:
            visit["bouts"] = _bouts_in_visit(visit, bs, be, peak, float(peak_speed_threshold_mm_s))
    return scenes, meta


def _aggressive_chaser_index(run_group) -> tuple[int, str]:
    """The chasing object's index from the CRA endpoint role codes -- never index order.
    Falls back to chaser 0 with a note if no endpoint is present."""

    parent = run_group.get("cra_primary_endpoint")
    if parent is not None and len(list(parent.keys())):
        cra = parent[sorted(parent.keys())[-1]]
        objs = cra.get("objects")
        if objs is not None and "object_role_code" in objs:
            idx = np.asarray(objs["object_index"][:]).reshape(-1)
            code = np.asarray(objs["object_role_code"][:]).reshape(-1)
            hit = np.flatnonzero(code == 1)
            if hit.size:
                return int(idx[int(hit[0])]), "cra_role_code"
    return 0, "fallback_chaser_0_no_cra_endpoint"


def collect_chase_ring_entries(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    swim_bout_run: str = "latest",
    bout_response_component: str = "latest",
    peak_speed_threshold_mm_s: float = DEFAULT_PEAK_SPEED_THRESHOLD_MM_S,
    epoch_label: str = "training_event",
    visit_enter_mm: float = 15.0,
    visit_exit_mm: float = 20.0,
    pad_s: float = 1.0,
    min_speed_mm_s: float = 0.5,
) -> tuple[list[VisitScene], dict[str, Any]]:
    """The training-epoch analogue of collect_ring_entries, in a CHASER-CENTRIC frame.

    During the chase the object MOVES, so the static-object ring frame is a fiction. Here the
    (moving) chaser is fixed at the origin and the fish is drawn relative to it, rotated so the
    chaser's direction of pursuit points +y -- the same frame chaser_escape_freeze scores the
    escape/freeze response in. The rings are literal distance-to-chaser bands; an entry is the
    fish crossing into 15 mm and not leaving until it passes 20 mm of the pursuer. There is no
    fixed wall arc, because there is no fixed anything but the chaser.
    """

    root = _open_root(zarr_path, mode="r")
    run_group, run_name, _run_path = _resolve_chaser_distance_run(root, chaser_distance_run)
    ppm = _safe_float(run_group.attrs.get("pixels_per_mm_projector"))
    fps = _safe_float(run_group.attrs.get("fps"), 100.0)

    agg_idx, role_source = _aggressive_chaser_index(run_group)
    chaser_indices = np.asarray(run_group["chasers"]["chaser_index"][:], dtype=np.int64).reshape(-1)
    col = int(np.flatnonzero(chaser_indices == agg_idx)[0]) if np.any(chaser_indices == agg_idx) else 0

    positions = run_group["positions"]
    fish_px = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float64)
    chaser_px = np.asarray(positions["chaser_arena_xy"][:, col, :], dtype=np.float64)
    fish_valid = np.asarray(positions["fish_valid"][:], dtype=bool)
    chaser_valid = np.asarray(positions["chaser_valid"][:, col], dtype=bool)
    total_frames = int(fish_px.shape[0])

    # chaser heading (direction of pursuit) and the fish in the chaser-centric frame
    heading_rad, _held, _ch_speed = _heading_angles_from_chaser(
        chaser_px, chaser_valid, ppm=float(ppm), fps=float(fps), min_speed_mm_s=float(min_speed_mm_s)
    )
    frame_xy, radius_mm, _bearing = chaser_frame_transform(
        fish_px, chaser_px, heading_rad, pixels_per_mm=float(ppm)
    )
    frame_xy = np.asarray(frame_xy, dtype=np.float64)      # (n, 2) mm, y-up, chaser heading -> +y
    dist = np.asarray(radius_mm, dtype=np.float64)
    frame_valid = fish_valid & chaser_valid & np.isfinite(dist) & np.isfinite(frame_xy).all(axis=1)

    # the training epoch window, settle-trimmed the same way every other component trims it
    epochs = _read_epochs(run_group, total_frames=total_frames)
    epochs = _apply_settle_trim(
        epochs,
        chaser_xy=np.asarray(positions["chaser_arena_xy"][:], dtype=np.float32),
        chaser_valid=np.asarray(positions["chaser_valid"][:], dtype=bool),
        fps=float(fps),
        pixels_per_mm=float(ppm),
        settle_trim_s=_protocol_position_transition_s(root, run_group),
        motion_spread_threshold_mm=1.0,
    )
    epoch = next((e for e in epochs if e.label == epoch_label), None)
    if epoch is None:
        raise ValueError(f"No {epoch_label!r} epoch in {run_name}.")

    lo, hi = int(epoch.start_frame), int(epoch.end_frame)
    slc = slice(lo, hi + 1)
    d_ep = dist[slc]
    ok_ep = frame_valid[slc]
    vid = _segment_visits(d_ep, ok_ep, enter_mm=float(visit_enter_mm), exit_mm=float(visit_exit_mm))

    label = run_group["chasers"].get("behavior_class_label_bytes")
    ref_label = "aggressive chaser"
    if label is not None:
        names = _decode_text_column(np.asarray(label[:]))
        if col < len(names) and names[col]:
            ref_label = f"{names[col]} (aggressive)"

    scene = VisitScene(epoch.label, ref_label, True, math.nan, math.nan)
    scene.chaser_centric = True
    pad = int(max(0.0, float(pad_s)) * fps)

    for v in range(int(vid.max()) + 1 if vid.max() >= 0 else 0):
        idx = np.flatnonzero(vid == v)
        if idx.size < 5:
            continue
        w0 = max(0, int(idx[0]) - pad)
        w1 = min(len(d_ep) - 1, int(idx[-1]) + pad)
        window = np.arange(w0, w1 + 1)
        window = window[ok_ep[window]]
        if window.size < 5:
            continue
        g = window + lo
        xy = frame_xy[g]
        # heading arrow = the fish's motion direction in the DRAWN frame, so it is always
        # consistent with the trajectory regardless of frame convention (the drawing code
        # negates y, so store atan2(-dy, dx) of the drawn deltas).
        head = np.full(xy.shape[0], np.nan)
        if xy.shape[0] >= 3:
            dxy = np.gradient(xy, axis=0)
            head = np.degrees(np.arctan2(-dxy[:, 1], dxy[:, 0]))
        scene.visits.append(
            {
                "visit_id": v,
                "xy": xy,
                "distance_mm": dist[g],
                "t_s": (window - idx[0]) / fps,
                "heading_deg": head,
                "cpa_mm": float(np.nanmin(dist[g])),
                "is_bout_onset": np.zeros(xy.shape[0], dtype=bool),
                "frames": g,
            }
        )

    bs, be, peak = _load_bout_segments(zarr_path, chaser_distance_run, bout_response_component)
    for visit in scene.visits:
        visit["bouts"] = _bouts_in_visit(visit, bs, be, peak, float(peak_speed_threshold_mm_s))

    meta = {
        "recording_id": str(run_group.attrs.get("recording_id") or Path(zarr_path).stem),
        "run": run_name,
        "fps": float(fps),
        "peak_speed_threshold_mm_s": float(peak_speed_threshold_mm_s),
        "responsive_band_mm": list(RESPONSIVE_BAND_MM),
        "ring_edges_mm": list(RING_EDGES_MM),
        "frame": "chaser_centric",
        "aggressive_chaser_index": int(agg_idx),
        "aggressive_role_source": role_source,
    }
    return [scene], meta


def _load_bout_segments(
    zarr_path: Path, chaser_distance_run: str, component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(start_frame, end_frame, peak_speed_mm_s) from the chaser_bout_response bout table.

    That table is the single source of both the bout boundaries and the peak speed used to
    call escapes, so the segments here and the escape counts in chaser_escape_events cannot
    drift apart. If it is absent, return empty arrays and the animation shows onsets only.
    """

    root = _open_root(zarr_path, mode="r")
    run_group, _name, _path = _resolve_chaser_distance_run(root, chaser_distance_run)
    parent = run_group.get("chaser_bout_response")
    if parent is None or not len(list(parent.keys())):
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64))
    name = str(component).strip()
    if not name or name == "latest":
        name = sorted(parent.keys())[-1]
    if name not in parent:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64))
    bouts = parent[name]["bouts"]
    valid = np.asarray(bouts["valid"][:], dtype=bool)
    bs = np.asarray(bouts["start_frame"][:], dtype=np.int64)
    be = np.asarray(bouts["end_frame"][:], dtype=np.int64)
    peak = np.asarray(bouts["peak_speed_mm_s"][:], dtype=np.float64)
    return bs[valid], be[valid], peak[valid]


def _bouts_in_visit(
    visit: dict[str, Any], bs: np.ndarray, be: np.ndarray, peak: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    frames = np.asarray(visit["frames"], dtype=np.int64)
    xy = np.asarray(visit["xy"], dtype=np.float64)
    if frames.size == 0 or bs.size == 0:
        return []
    lo, hi = int(frames[0]), int(frames[-1])
    out: list[dict[str, Any]] = []
    # bouts whose onset falls within this entry's window
    for k in np.flatnonzero((bs >= lo) & (bs <= hi)):
        gs, ge = int(bs[k]), int(be[k])
        seg = np.flatnonzero((frames >= gs) & (frames <= ge))
        if seg.size < 2:
            continue
        i0, i1 = int(seg[0]), int(seg[-1])
        out.append(
            {
                "xy": xy[i0 : i1 + 1],
                "peak_speed_mm_s": float(peak[k]),
                "is_escape": bool(peak[k] > threshold),
                "i0": i0,
                "i1": i1,
            }
        )
    out.sort(key=lambda b: b["i0"])
    return out


# ==========================================================================================
# The ring frame
# ==========================================================================================


def _draw_rings(ax, scene: VisitScene, *, limit: float) -> None:
    """Object at origin, concentric responsive rings, the 8-16 mm shell shaded, wall arc."""

    # responsive shell first, so everything else sits on top
    inner, outer = RESPONSIVE_BAND_MM
    ax.add_patch(plt.Circle((0.0, 0.0), outer, color=RESP_C, alpha=0.10, zorder=0))
    ax.add_patch(plt.Circle((0.0, 0.0), inner, color="white", zorder=0))
    for r in RING_EDGES_MM:
        in_shell = inner <= r <= outer
        ax.add_patch(plt.Circle((0.0, 0.0), r, fill=False, lw=1.0 if in_shell else 0.7,
                                 color=RESP_C if in_shell else "#cbd5e1",
                                 ls="-" if in_shell else ":", zorder=1))
    # the measured escape trigger radius
    ax.add_patch(plt.Circle((0.0, 0.0), ESCAPE_TRIGGER_MM, fill=False, lw=1.0, ls="--",
                            color=ESCAPE_C, alpha=0.55, zorder=1))
    chaser_centric = bool(getattr(scene, "chaser_centric", False))
    if chaser_centric:
        # no fixed wall to draw -- the chaser moves. Mark the pursuit direction instead.
        ax.annotate("", xy=(0.0, limit * 0.9), xytext=(0.0, limit * 0.72),
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.4))
        ax.text(0.0, limit * 0.93, "pursuit", ha="center", va="bottom", fontsize=7, color="#64748b")
    elif np.isfinite(scene.arena_center_distance_mm) and np.isfinite(scene.arena_radius_mm):
        # the wall, where it really falls in this static-object frame
        ax.add_patch(plt.Circle((scene.arena_center_distance_mm, 0.0), scene.arena_radius_mm,
                                fill=False, color="#64748b", lw=1.6, zorder=1))
    # the object / chaser
    ax.add_patch(plt.Circle((0.0, 0.0), DOT_RADIUS_MM,
                            color=ESCAPE_C if scene.is_object else "#94a3b8",
                            alpha=0.9 if scene.is_object else 0.4, zorder=5))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_bout_segment(ax, bout: dict[str, Any], *, alpha: float = 1.0, zorder: int = 4) -> None:
    xy = bout["xy"]
    if bout["is_escape"]:
        ax.plot(xy[:, 0], xy[:, 1], color=ESCAPE_C, lw=2.6, alpha=alpha, solid_capstyle="round",
                zorder=zorder + 1)
        ax.plot(xy[-1, 0], xy[-1, 1], marker="*", ms=11, color=ESCAPE_C, alpha=alpha, zorder=zorder + 2)
    else:
        ax.plot(xy[:, 0], xy[:, 1], color=BOUT_C, lw=1.7, alpha=alpha, solid_capstyle="round",
                zorder=zorder)
        ax.plot(xy[0, 0], xy[0, 1], marker=".", ms=5, color=BOUT_C, alpha=alpha, zorder=zorder)


# ==========================================================================================
# Entry ordering: surface the legible, interesting entries first
# ==========================================================================================


def _entry_sort_key(row: tuple[VisitScene, dict[str, Any]]) -> tuple:
    """Escape-containing entries first, then closer approaches, then shorter (more legible)
    ones. A wall-dwelling fish produces a few 200-bout lingering entries that swamp the
    genuine approach-and-leave passes; this floats the readable ones to the top."""

    scene, visit = row
    bouts = visit.get("bouts", [])
    n_esc = sum(int(b["is_escape"]) for b in bouts)
    return (-n_esc, float(visit.get("cpa_mm", math.inf)), len(visit["xy"]))


def _ordered_entries(scenes: Sequence[VisitScene], object_only: bool) -> list[tuple[VisitScene, dict]]:
    rows = [(s, v) for s in scenes if (s.is_object or not object_only) for v in s.visits]
    return sorted(rows, key=_entry_sort_key)


# ==========================================================================================
# Static figure: every entry, bouts drawn through the rings
# ==========================================================================================


def render_ring_entries_png(
    scenes: Sequence[VisitScene], meta: dict[str, Any], *, object_only: bool = True,
    limit: float = 24.0, max_panels: int = 24,
) -> bytes:
    rows = _ordered_entries(scenes, object_only)
    if not rows:
        raise ValueError("No entries to render.")
    shown = rows[:max_panels]
    n_cols = 6
    n_rows = int(math.ceil(len(shown) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.9 * n_rows), squeeze=False)
    for k, (scene, visit) in enumerate(shown):
        ax = axes[k // n_cols][k % n_cols]
        _draw_rings(ax, scene, limit=limit)
        xy = visit["xy"]
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", lw=0.8, alpha=0.9, zorder=2)   # coast path
        n_esc = 0
        for bout in visit.get("bouts", []):
            _draw_bout_segment(ax, bout)
            n_esc += int(bout["is_escape"])
        ax.plot(xy[0, 0], xy[0, 1], marker="s", ms=5, color="#16a34a", zorder=6)     # entry
        j = int(np.argmin(visit["distance_mm"]))
        ax.plot(xy[j, 0], xy[j, 1], marker="o", ms=7, mfc="none", mec="#b91c1c", mew=1.4, zorder=6)
        nb = len(visit.get("bouts", []))
        ax.set_title(f"{scene.epoch_label.replace('_',' ')} · entry {visit['visit_id']}\n"
                     f"CPA {visit['cpa_mm']:.1f} mm · {nb} bouts"
                     + (f" · {n_esc} escape" if n_esc else ""),
                     fontsize=7, color=ESCAPE_C if n_esc else "#334155")
    for k in range(len(shown), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    dropped = len(rows) - len(shown)
    extra = f"   (showing {len(shown)} of {len(rows)} entries)" if dropped else ""
    is_chase = meta.get("frame") == "chaser_centric"
    frame_line = (
        "MOVING chaser at origin, rotated so pursuit points up · no fixed wall (nothing is fixed but the chaser)"
        if is_chase else
        "static object at origin, arena centre rotated to +x · grey arc = wall"
    )
    fig.suptitle(
        f"Bouts per entry through the responsive rings — {meta['recording_id']}{extra}\n"
        f"amber shell = 8–16 mm responsive band · red dashed = {ESCAPE_TRIGGER_MM:g} mm escape trigger · "
        f"{frame_line}\n"
        "green square = entry · blue = ordinary bout · red ★ = escape bout · red circle = closest approach",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


# ==========================================================================================
# Animation: fish traverses the rings, bouts accumulate per entry
# ==========================================================================================


def write_ring_traversal_gif(
    scenes: Sequence[VisitScene], meta: dict[str, Any], out_path: Path, *,
    object_only: bool = True, limit: float = 24.0, fps: int = 25,
    max_entries: int = 12, frames_per_entry: int = 90, trail_frac: float = 0.25,
    hold_s: float = 0.7,
) -> Path:
    """Replay each entry: the fish moves through the rings and each bout, once executed, stays
    lit for the rest of that entry -- so the *set* of bouts builds up as you watch.

    Each entry is bound to ``frames_per_entry`` display frames regardless of how long the fish
    actually lingered, so a 4 s approach and a 130 s wall-dwell take the same wall-clock. Long
    entries are subsampled (real xy indices are kept, so bout onsets still register); short
    ones play at full detail. Entries are ordered by _entry_sort_key and capped at
    ``max_entries`` -- the legible, escape-containing passes lead."""

    rows = _ordered_entries(scenes, object_only)[:max_entries]
    if not rows:
        raise ValueError("No entries to animate.")

    hold_frames = max(1, int(hold_s * fps))
    timeline: list[tuple[int, int, bool]] = []       # (row, xy-index, is_hold)
    for r_idx, (_scene, visit) in enumerate(rows):
        n = len(visit["xy"])
        idx = np.unique(np.linspace(0, n - 1, min(n, frames_per_entry)).astype(int))
        for f in idx:
            timeline.append((r_idx, int(f), False))
        for _ in range(hold_frames):                 # pause on the completed entry
            timeline.append((r_idx, n - 1, True))

    fig, ax = plt.subplots(figsize=(6.0, 6.4))

    def draw(i: int) -> None:
        r_idx, f, is_hold = timeline[i]
        scene, visit = rows[r_idx]
        trail_n = max(2, int(trail_frac * len(visit["xy"])))
        ax.clear()
        _draw_rings(ax, scene, limit=limit)
        xy = visit["xy"]
        bouts = visit.get("bouts", [])

        # coast path up to now, faint
        ax.plot(xy[: f + 1, 0], xy[: f + 1, 1], color="#e2e8f0", lw=0.8, zorder=2)
        # every bout already begun by frame f, lit and persisting
        n_done = 0
        n_esc = 0
        for bout in bouts:
            if bout["i0"] <= f:
                clipped = dict(bout)
                clipped["xy"] = bout["xy"][: max(2, min(bout["i1"], f) - bout["i0"] + 1)]
                _draw_bout_segment(ax, clipped, alpha=1.0 if bout["i1"] <= f else 0.85)
                if bout["i1"] <= f:
                    n_done += 1
                    n_esc += int(bout["is_escape"])
        # the fish
        ax.plot(xy[max(0, f - trail_n):f + 1, 0], xy[max(0, f - trail_n):f + 1, 1],
                color="#1d4ed8", lw=1.2, alpha=0.4, zorder=3)
        ax.plot(xy[f, 0], xy[f, 1], marker="o", ms=8, color="#0f172a", zorder=7)
        h = visit["heading_deg"][f]
        if np.isfinite(h):
            ax.arrow(xy[f, 0], xy[f, 1], 3.2 * math.cos(math.radians(h)),
                     -3.2 * math.sin(math.radians(h)), head_width=1.2, color="#0f172a",
                     zorder=7, length_includes_head=True)
        j = int(np.argmin(visit["distance_mm"]))
        if f >= j:
            ax.plot(xy[j, 0], xy[j, 1], marker="o", ms=9, mfc="none", mec="#b91c1c", mew=1.6, zorder=6)

        d = float(visit["distance_mm"][f])
        in_shell = RESPONSIVE_BAND_MM[0] <= d <= RESPONSIVE_BAND_MM[1]
        esc_tag = f"  ·  {n_esc} escape" if n_esc else ""
        ax.set_title(
            f"{meta['recording_id']}  ·  {scene.epoch_label.replace('_',' ')}  ·  entry {visit['visit_id']}\n"
            f"d = {d:5.1f} mm {'‹ in responsive shell ›' if in_shell else ''}\n"
            f"bouts this entry: {n_done}/{len(bouts)}{esc_tag}",
            fontsize=9, color=ESCAPE_C if n_esc else ("#b45309" if in_shell else "#334155"),
        )

    anim = FuncAnimation(fig, draw, frames=len(timeline), interval=1000 // fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400) if out_path.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zarr_path", type=Path)
    p.add_argument("--chaser-distance-run", default="latest")
    p.add_argument("--swim-bout-run", default="latest")
    p.add_argument("--epochs", default=",".join(DEFAULT_EPOCHS))
    p.add_argument("--virtual-rotations-deg", default=",".join(f"{v:g}" for v in DEFAULT_VIRTUAL_ROTATIONS_DEG))
    p.add_argument("--limit-mm", type=float, default=24.0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--gif", type=Path, default=None, help="Write the traversal animation here (.mp4 or .gif).")
    p.add_argument("--gif-include-virtual", action="store_true")
    p.add_argument("--gif-fps", type=int, default=25)
    p.add_argument("--gif-max-entries", type=int, default=12)
    p.add_argument("--gif-frames-per-entry", type=int, default=90,
                   help="Display frames per entry; long lingering entries are subsampled to fit.")
    return p


CHASE_EPOCHS = ("training_event", "chase", "chase_event")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    wanted = [e.strip() for e in str(args.epochs).split(",") if e.strip()]
    chase_wanted = [e for e in wanted if e in CHASE_EPOCHS]
    static_wanted = [e for e in wanted if e not in CHASE_EPOCHS]

    # The chase epoch's chaser moves, so it needs the chaser-centric frame. A request that mixes
    # chase and static epochs cannot share one frame; keep this call to a single kind.
    if chase_wanted and static_wanted:
        raise SystemExit(
            "The training/chase epoch uses a chaser-centric frame and cannot be combined with "
            "static epochs in one figure. Run them separately, e.g. --epochs training_event."
        )

    if chase_wanted:
        scenes, meta = collect_chase_ring_entries(
            Path(args.zarr_path),
            chaser_distance_run=str(args.chaser_distance_run),
            swim_bout_run=str(args.swim_bout_run),
            epoch_label=chase_wanted[0],
        )
    else:
        scenes, meta = collect_ring_entries(
            Path(args.zarr_path),
            chaser_distance_run=str(args.chaser_distance_run),
            swim_bout_run=str(args.swim_bout_run),
            epochs_wanted=static_wanted,
            virtual_rotations_deg=[float(v) for v in str(args.virtual_rotations_deg).split(",") if v.strip()],
        )
    for scene in scenes:
        if not scene.is_object:
            continue
        nb = sum(len(v.get("bouts", [])) for v in scene.visits)
        ne = sum(int(b["is_escape"]) for v in scene.visits for b in v.get("bouts", []))
        print(f"{scene.epoch_label:<12} {scene.ref_label:<26} entries={len(scene.visits):<3} "
              f"bouts={nb:<4} escapes={ne}")

    epoch_tag = (chase_wanted[0] if chase_wanted else "_".join(static_wanted)) or "epochs"
    out_dir = args.out_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{meta['recording_id']}_ring_entries_{epoch_tag}.png"
    png.write_bytes(render_ring_entries_png(scenes, meta, limit=float(args.limit_mm)))
    print(f"wrote {png}")

    if args.gif:
        path = write_ring_traversal_gif(
            scenes, meta, Path(args.gif), object_only=not bool(args.gif_include_virtual),
            limit=float(args.limit_mm), fps=int(args.gif_fps),
            max_entries=int(args.gif_max_entries), frames_per_entry=int(args.gif_frames_per_entry),
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
