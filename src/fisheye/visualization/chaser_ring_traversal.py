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

from fisheye.analysis.chaser_escape_events import DEFAULT_PEAK_SPEED_THRESHOLD_MM_S
from fisheye.analysis.chaser_radial_occupancy import (
    _open_root,
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
    # the wall, where it really falls in this frame
    ax.add_patch(plt.Circle((scene.arena_center_distance_mm, 0.0), scene.arena_radius_mm,
                            fill=False, color="#64748b", lw=1.6, zorder=1))
    # the object
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
    fig.suptitle(
        f"Bouts per entry through the responsive rings — {meta['recording_id']}{extra}\n"
        f"amber shell = 8–16 mm responsive band · red dashed = {ESCAPE_TRIGGER_MM:g} mm escape trigger · "
        "grey arc = wall\n"
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenes, meta = collect_ring_entries(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        swim_bout_run=str(args.swim_bout_run),
        epochs_wanted=[e.strip() for e in str(args.epochs).split(",") if e.strip()],
        virtual_rotations_deg=[float(v) for v in str(args.virtual_rotations_deg).split(",") if v.strip()],
    )
    for scene in scenes:
        if not scene.is_object:
            continue
        nb = sum(len(v.get("bouts", [])) for v in scene.visits)
        ne = sum(int(b["is_escape"]) for v in scene.visits for b in v.get("bouts", []))
        print(f"{scene.epoch_label:<12} {scene.ref_label:<26} entries={len(scene.visits):<3} "
              f"bouts={nb:<4} escapes={ne}")

    out_dir = args.out_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{meta['recording_id']}_ring_entries.png"
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
