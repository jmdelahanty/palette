"""Plot and animate the fish's trajectory during each near-object visit.

With only 4-10 approach events per epoch, a median over them is noise. The honest analysis
is to look at the trajectories. This renders them.

CANONICAL FRAME
---------------
Every visit is drawn with the **object at the origin and the arena centre rotated onto +x**.
Two things follow, and both matter:

* The objects sit at *different* arena positions in pre and post, so raw arena coordinates
  would scatter the pre and post visits into different corners and make them incomparable.
  Object-centred coordinates put them on top of each other.
* The arena wall lands in the **same place in every panel**. Since the objects sit ~7 mm
  from the wall, that arc is the confound, drawn: a wall-following fish must sweep past the
  object whether or not it cares about it.

VIRTUAL CONTROLS
----------------
Each object is rendered beside its **virtual twins** -- the same position rotated about the
arena centre, i.e. an empty patch of arena with identical wall geometry. If the fish's path
near the real dot looks like its path near an empty spot, there is no object-driven
behavior, and the eye will see that faster than any statistic.

Use `--gif` for an animation replaying each visit frame by frame.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402

from fisheye.analysis.chaser_bout_response import (
    DEFAULT_VISIT_ENTER_MM,
    DEFAULT_VISIT_EXIT_MM,
    _resolve_heading,
    _segment_visits,
)
from fisheye.analysis.swim_bout_io import resolve_swim_bout_run_name
from fisheye.analysis.chaser_radial_occupancy import (
    _apply_settle_trim,
    _open_root,
    _protocol_position_transition_s,
    _read_epochs,
    _resolve_arena_geometry,
    _resolve_chaser_distance_run,
)


DEFAULT_EPOCHS = ("pre_event", "post_event")
DEFAULT_VIRTUAL_ROTATIONS_DEG = (120.0, 240.0)
DOT_RADIUS_MM = 2.0


class VisitScene:
    """One (epoch, reference) cell: the reference's canonical frame and its visits."""

    def __init__(self, epoch_label: str, ref_label: str, is_object: bool,
                 arena_center_distance_mm: float, arena_radius_mm: float):
        self.epoch_label = epoch_label
        self.ref_label = ref_label
        self.is_object = is_object
        self.arena_center_distance_mm = arena_center_distance_mm
        self.arena_radius_mm = arena_radius_mm
        self.visits: list[dict[str, Any]] = []


def _canonicalize(fish_mm: np.ndarray, ref_mm: np.ndarray, center_mm: np.ndarray) -> tuple[np.ndarray, float]:
    """Object -> origin, arena centre -> +x. Returns the transformed fish path and the
    object's distance from the arena centre (where the centre lands on +x)."""

    to_center = center_mm - ref_mm
    d_c = float(np.hypot(to_center[0], to_center[1]))
    theta = math.atan2(float(to_center[1]), float(to_center[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    rel = np.asarray(fish_mm, dtype=np.float64) - np.asarray(ref_mm, dtype=np.float64)
    out = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)
    return out, d_c


def _rotate_about(point: np.ndarray, center: np.ndarray, degrees: float) -> np.ndarray:
    t = math.radians(float(degrees))
    c, s = math.cos(t), math.sin(t)
    rel = np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return np.asarray([rel[0] * c - rel[1] * s, rel[0] * s + rel[1] * c]) + center


def collect_visits(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    swim_bout_run: str = "latest",
    epochs_wanted: Sequence[str] = DEFAULT_EPOCHS,
    virtual_rotations_deg: Sequence[float] = DEFAULT_VIRTUAL_ROTATIONS_DEG,
    visit_enter_mm: float = DEFAULT_VISIT_ENTER_MM,
    visit_exit_mm: float = DEFAULT_VISIT_EXIT_MM,
    pad_s: float = 1.0,
) -> tuple[list[VisitScene], dict[str, Any]]:
    """Fail closed until every derived input to the visit view is sealed."""

    root = _open_root(zarr_path, mode="r")
    distance, _run_name, _run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    distance.require_derived_surface_authority("egocentric_bearing")


def _collect_visits_unsealed_inspection(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    swim_bout_run: str = "latest",
    epochs_wanted: Sequence[str] = DEFAULT_EPOCHS,
    virtual_rotations_deg: Sequence[float] = DEFAULT_VIRTUAL_ROTATIONS_DEG,
    visit_enter_mm: float = DEFAULT_VISIT_ENTER_MM,
    visit_exit_mm: float = DEFAULT_VISIT_EXIT_MM,
    pad_s: float = 1.0,
    egocentric_dependency_handle: dict[str, Any] | None = None,
    swim_bout_legacy_compatibility: bool = False,
) -> tuple[list[VisitScene], dict[str, Any]]:
    root = _open_root(zarr_path, mode="r")
    distance, run_name, run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    ppm = float(distance.pixels_per_mm_projector)
    fps = float(distance.fps)
    geometry = _resolve_arena_geometry(
        root,
        distance,
        pixels_per_mm=float(ppm),
    )
    if geometry.shape != "circle" or geometry.center_x_px is None:
        raise ValueError("Visit trajectories need a circular arena geometry.")

    fish = np.asarray(distance.fish_centroid_arena_xy, dtype=np.float64) / ppm
    chaser = np.asarray(distance.chaser_arena_xy, dtype=np.float64) / ppm
    fish_valid = np.asarray(distance.fish_valid, dtype=bool)
    chaser_valid = np.asarray(distance.chaser_valid, dtype=bool)
    chaser_idx = np.asarray(distance.chaser_index, dtype=np.int64).reshape(-1)
    labels: tuple[str, ...] = ()

    center = np.asarray([geometry.center_x_px, geometry.center_y_px], dtype=np.float64) / ppm
    arena_r = float(geometry.radius_px) / ppm
    total_frames = int(fish.shape[0])

    all_epochs = _read_epochs(distance, total_frames=total_frames)
    all_epochs = _apply_settle_trim(
        all_epochs,
        chaser_xy=np.asarray(distance.chaser_arena_xy, dtype=np.float32),
        chaser_valid=chaser_valid,
        fps=float(fps),
        pixels_per_mm=float(ppm),
        settle_trim_s=_protocol_position_transition_s(root, distance),
        motion_spread_threshold_mm=1.0,
    )

    # Explicit candidate handles allow inspection of sealed selector-ineligible
    # components without weakening ordinary selector resolution.
    heading, _heading_valid, resolved_ego, _ego_manifest_sha256 = _resolve_heading(
        root,
        snapshot=distance,
        run_group=root[run_path],
        total_frames=total_frames,
        dependency_handle=egocentric_dependency_handle,
    )

    # bout onsets, to mark where the fish re-aims
    bout_start = np.zeros(0, dtype=np.int64)
    swim_bout_name = resolve_swim_bout_run_name(
        root,
        run_name=swim_bout_run,
        legacy_compatibility=swim_bout_legacy_compatibility,
    )
    swim_bout_group = root[f"analysis/swim_bout_runs/{swim_bout_name}"]
    tables = swim_bout_group.get("tables")
    if tables is None or "bouts" not in tables:
        raise ValueError(
            "Visit trajectories require a canonical compact swim-bout table."
        )
    bout_table = tables["bouts"]
    if "start_frame" not in bout_table:
        raise ValueError("Selected swim-bout table lacks start_frame.")
    bout_start = np.asarray(
        bout_table["start_frame"][:],
        dtype=np.int64,
    ).reshape(-1)

    pad = int(max(0.0, float(pad_s)) * fps)
    scenes: list[VisitScene] = []

    for epoch in all_epochs:
        if epoch.label not in epochs_wanted:
            continue
        slc = slice(epoch.start_frame, epoch.end_frame + 1)
        f_xy = fish[slc]
        f_ok = fish_valid[slc] & np.isfinite(f_xy).all(axis=1)

        for c_idx in range(int(chaser_idx.shape[0])):
            base_label = labels[c_idx] if c_idx < len(labels) and labels[c_idx] else f"chaser{int(chaser_idx[c_idx])}"
            c_xy = chaser[slc, c_idx, :]
            ok_c = chaser_valid[slc, c_idx] & np.isfinite(c_xy).all(axis=1)
            if not np.any(ok_c):
                continue
            ref_static = np.nanmedian(c_xy[ok_c], axis=0)  # static in pre/post

            refs: list[tuple[str, np.ndarray, bool]] = [(base_label, ref_static, True)]
            for rot in virtual_rotations_deg:
                refs.append((f"virtual {base_label} {int(rot)}deg", _rotate_about(ref_static, center, rot), False))

            for ref_label, ref_pos, is_object in refs:
                canon, d_c = _canonicalize(f_xy, ref_pos, center)
                dist = np.hypot(f_xy[:, 0] - ref_pos[0], f_xy[:, 1] - ref_pos[1])
                vid = _segment_visits(dist, f_ok, enter_mm=visit_enter_mm, exit_mm=visit_exit_mm)
                scene = VisitScene(epoch.label, ref_label, is_object, d_c, arena_r)

                for v in range(int(vid.max()) + 1 if vid.max() >= 0 else 0):
                    idx = np.where(vid == v)[0]
                    if idx.size < 5:
                        continue
                    lo = max(0, int(idx[0]) - pad)
                    hi = min(len(canon) - 1, int(idx[-1]) + pad)
                    window = np.arange(lo, hi + 1)
                    window = window[f_ok[window]]
                    if window.size < 5:
                        continue
                    g = window + epoch.start_frame
                    scene.visits.append(
                        {
                            "visit_id": v,
                            "xy": canon[window],
                            "distance_mm": dist[window],
                            "t_s": (window - idx[0]) / fps,
                            # arena heading is y-down when drawn as (cos h, -sin h); the
                            # canonical frame rotates positions by -theta, so the heading angle
                            # must rotate by +theta to stay aligned with the drawn trajectory.
                            # (Subtracting it, the earlier bug, left the arrow ~perpendicular.)
                            "heading_deg": heading[g] + math.degrees(
                                math.atan2(*(center - ref_pos)[::-1])
                            ),
                            "cpa_mm": float(np.nanmin(dist[idx])),
                            "is_bout_onset": np.isin(g, bout_start),
                            "frames": g,
                        }
                    )
                scenes.append(scene)

    meta = {
        "recording_id": distance.recording_id,
        "run": run_name,
        "fps": float(fps),
        "arena_radius_mm": arena_r,
        "visit_enter_mm": float(visit_enter_mm),
        "visit_exit_mm": float(visit_exit_mm),
    }
    return scenes, meta


def _draw_frame(ax, scene: VisitScene, *, limit: float) -> None:
    """Object at origin, arena centre on +x, wall as the arc it really is."""

    wall = plt.Circle((scene.arena_center_distance_mm, 0.0), scene.arena_radius_mm,
                      fill=False, color="#64748b", lw=1.6, ls="-", zorder=1)
    ax.add_patch(wall)
    ax.add_patch(plt.Circle((0.0, 0.0), DOT_RADIUS_MM,
                            color="#dc2626" if scene.is_object else "#94a3b8",
                            alpha=0.9 if scene.is_object else 0.45, zorder=4))
    ax.add_patch(plt.Circle((0.0, 0.0), scene.arena_center_distance_mm * 0 + 15.0,
                            fill=False, color="#cbd5e1", lw=0.8, ls=":", zorder=1))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_overlay_png(scenes: Sequence[VisitScene], meta: dict[str, Any], *, limit: float = 22.0) -> bytes:
    """All visits for each (epoch, reference), overlaid, coloured by time within the visit."""

    epochs = sorted({s.epoch_label for s in scenes}, key=lambda e: 0 if "pre" in e else 1)
    ref_order = [s.ref_label for s in scenes if s.epoch_label == epochs[0]]
    seen: list[str] = []
    for r in ref_order:
        if r not in seen:
            seen.append(r)

    fig, axes = plt.subplots(len(epochs), len(seen),
                             figsize=(3.1 * len(seen), 3.4 * len(epochs)), squeeze=False)
    for i, epoch in enumerate(epochs):
        for j, ref in enumerate(seen):
            ax = axes[i][j]
            match = [s for s in scenes if s.epoch_label == epoch and s.ref_label == ref]
            if not match:
                ax.axis("off")
                continue
            scene = match[0]
            _draw_frame(ax, scene, limit=limit)
            for visit in scene.visits:
                xy = visit["xy"]
                pts = xy.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap="viridis",
                                    norm=plt.Normalize(0, max(len(xy) - 1, 1)),
                                    linewidth=1.4, alpha=0.9, zorder=3)
                lc.set_array(np.arange(len(xy) - 1))
                ax.add_collection(lc)
                k = int(np.argmin(visit["distance_mm"]))
                ax.plot(xy[k, 0], xy[k, 1], marker="o", ms=4.5, mfc="none",
                        mec="#b91c1c" if scene.is_object else "#475569", mew=1.2, zorder=5)
            title = f"{ref}\n{len(scene.visits)} visits"
            ax.set_title(title, fontsize=8,
                         color="#b91c1c" if scene.is_object else "#64748b",
                         fontweight="bold" if scene.is_object else "normal")
            if j == 0:
                ax.text(-0.08, 0.5, epoch.replace("_", " "), transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle(
        f"Fish trajectory during each near-object visit — {meta['recording_id']}\n"
        "object (red) at origin · arena centre rotated onto +x · grey arc = the wall · "
        "circle = closest approach · colour = time\n"
        "grey dots are VIRTUAL controls: same wall geometry, no object",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.90))
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def render_per_visit_png(scenes: Sequence[VisitScene], meta: dict[str, Any], *,
                         object_only: bool = True, limit: float = 22.0) -> bytes:
    """One panel per visit -- the thing to actually look at when n is 6."""

    chosen = [s for s in scenes if (s.is_object or not object_only)]
    rows = [(s, v) for s in chosen for v in s.visits]
    if not rows:
        raise ValueError("No visits to render.")
    n_cols = 6
    n_rows = int(math.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.8 * n_rows), squeeze=False)
    for k, (scene, visit) in enumerate(rows):
        ax = axes[k // n_cols][k % n_cols]
        _draw_frame(ax, scene, limit=limit)
        xy = visit["xy"]
        pts = xy.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, max(len(xy) - 1, 1)),
                            linewidth=1.8, zorder=3)
        lc.set_array(np.arange(len(xy) - 1))
        ax.add_collection(lc)
        onsets = visit["is_bout_onset"]
        if np.any(onsets):
            ax.plot(xy[onsets, 0], xy[onsets, 1], ls="none", marker=".", ms=5,
                    color="#f59e0b", zorder=6, label="bout onset")
        ax.plot(xy[0, 0], xy[0, 1], marker="s", ms=4, color="#16a34a", zorder=6)
        j = int(np.argmin(visit["distance_mm"]))
        ax.plot(xy[j, 0], xy[j, 1], marker="o", ms=6, mfc="none", mec="#b91c1c", mew=1.4, zorder=6)
        ax.set_title(f"{scene.epoch_label.replace('_',' ')} · {scene.ref_label}\n"
                     f"visit {visit['visit_id']} · CPA {visit['cpa_mm']:.1f} mm",
                     fontsize=7)
    for k in range(len(rows), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.suptitle(
        f"Every near-object visit — {meta['recording_id']}\n"
        "green square = entry · red circle = closest approach · amber = bout onsets · "
        "grey arc = arena wall",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def write_gif(scenes: Sequence[VisitScene], meta: dict[str, Any], out_path: Path, *,
              object_only: bool = True, limit: float = 22.0, fps: int = 25,
              trail_s: float = 2.0, stride: int = 2) -> Path:
    """Replay every visit: fish, heading arrow, trail, closest-approach marker."""

    chosen = [s for s in scenes if (s.is_object or not object_only)]
    rows = [(s, v) for s in chosen for v in s.visits]
    if not rows:
        raise ValueError("No visits to animate.")

    timeline: list[tuple[int, int]] = []
    for r_idx, (_scene, visit) in enumerate(rows):
        for f in range(0, len(visit["xy"]), max(1, int(stride))):
            timeline.append((r_idx, f))

    fig, ax = plt.subplots(figsize=(5.4, 5.8))
    trail_n = max(2, int(trail_s * meta["fps"] / max(1, stride)))

    def draw(i: int) -> None:
        r_idx, f = timeline[i]
        scene, visit = rows[r_idx]
        ax.clear()
        _draw_frame(ax, scene, limit=limit)
        xy = visit["xy"]
        lo = max(0, f - trail_n * max(1, stride))
        ax.plot(xy[lo:f + 1, 0], xy[lo:f + 1, 1], color="#2563eb", lw=1.6, alpha=0.85, zorder=3)
        ax.plot(xy[f, 0], xy[f, 1], marker="o", ms=7, color="#1d4ed8", zorder=6)
        h = visit["heading_deg"][f]
        if np.isfinite(h):
            # heading is y-up CCW; the panel is y-down, so negate the y component
            ax.arrow(xy[f, 0], xy[f, 1], 3.0 * math.cos(math.radians(h)),
                     -3.0 * math.sin(math.radians(h)), head_width=1.1,
                     color="#1d4ed8", zorder=6, length_includes_head=True)
        j = int(np.argmin(visit["distance_mm"]))
        if f >= j:
            ax.plot(xy[j, 0], xy[j, 1], marker="o", ms=8, mfc="none", mec="#b91c1c", mew=1.5, zorder=5)
        ax.set_title(
            f"{meta['recording_id']}\n{scene.epoch_label.replace('_',' ')} · {scene.ref_label} · "
            f"visit {visit['visit_id']}\nd = {visit['distance_mm'][f]:5.1f} mm   "
            f"(CPA {visit['cpa_mm']:.1f} mm)",
            fontsize=9, color="#b91c1c" if scene.is_object else "#475569",
        )

    anim = FuncAnimation(fig, draw, frames=len(timeline), interval=1000 // fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4 is ~20x faster to encode and ~10x smaller than an equivalent gif; a full replay of
    # every visit at native rate is tens of thousands of frames, so use it when asked for.
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
    p.add_argument("--visit-enter-mm", type=float, default=DEFAULT_VISIT_ENTER_MM)
    p.add_argument("--visit-exit-mm", type=float, default=DEFAULT_VISIT_EXIT_MM)
    p.add_argument("--limit-mm", type=float, default=22.0)
    p.add_argument("--out-dir", type=Path, default=None, help="Write PNGs here.")
    p.add_argument("--gif", type=Path, default=None, help="Write an animation to this path.")
    p.add_argument("--gif-include-virtual", action="store_true")
    p.add_argument("--gif-fps", type=int, default=25)
    p.add_argument("--gif-stride", type=int, default=4,
                   help="Take every Nth camera frame. At 100 fps, 4 gives a 25 Hz replay.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenes, meta = collect_visits(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        swim_bout_run=str(args.swim_bout_run),
        epochs_wanted=[e.strip() for e in str(args.epochs).split(",") if e.strip()],
        virtual_rotations_deg=[float(v) for v in str(args.virtual_rotations_deg).split(",") if v.strip()],
        visit_enter_mm=float(args.visit_enter_mm),
        visit_exit_mm=float(args.visit_exit_mm),
    )
    for scene in scenes:
        kind = "OBJECT " if scene.is_object else "control"
        print(f"{kind} {scene.epoch_label:<12} {scene.ref_label:<28} visits={len(scene.visits)}")

    out_dir = args.out_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = out_dir / f"{meta['recording_id']}_visits_overlay.png"
    overlay.write_bytes(render_overlay_png(scenes, meta, limit=float(args.limit_mm)))
    print(f"wrote {overlay}")
    per_visit = out_dir / f"{meta['recording_id']}_visits_each.png"
    per_visit.write_bytes(render_per_visit_png(scenes, meta, limit=float(args.limit_mm)))
    print(f"wrote {per_visit}")

    if args.gif:
        path = write_gif(scenes, meta, Path(args.gif),
                         object_only=not bool(args.gif_include_virtual),
                         limit=float(args.limit_mm), fps=int(args.gif_fps),
                         stride=int(args.gif_stride))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
