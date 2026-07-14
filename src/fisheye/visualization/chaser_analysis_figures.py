"""Per-recording summary sheets and cohort figures for the chaser analysis stack.

Two entry points:

    render_recording_summary(zarr_path)   -> a one-page PNG per fish
    render_cohort_figures(registry, ...)  -> group-level figures across recordings

Design rules, learned the hard way on this dataset:

* **Always draw the control next to the effect.** The fish wall-follows and the objects sit
  near the wall, so an object curve alone is uninterpretable. Every object trace is plotted
  against its virtual twins (the same position rotated about the arena centre -- matched wall
  geometry, no object).
* **Show the dose-response, not a scalar.** A single near-band number averages over bins where
  nothing happens; on this cohort that halved the signal. A real object-driven effect is
  localized and decays with distance. An artifact is flat.
* **Show the n.** Bouts within one visit are not independent, so visit counts are annotated,
  and cohort panels plot every fish rather than only a mean.
* **Show the missing data.** Detector recall collapses when the fish freezes at the wall --
  precisely the behavior of interest -- so tracking dropout is on the sheet, not in a footnote.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path
import sqlite3
from typing import Any, Optional, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.shared.arena_geometry import resolve_arena_geometry  # noqa: E402

warnings.filterwarnings("ignore")

OBJ_COLOR = {"aggressive": "#dc2626", "inert": "#2563eb"}
CTRL_COLOR = "#94a3b8"
EPOCH_STYLE = {"pre_event": dict(ls="--", alpha=0.85), "post_event": dict(ls="-", alpha=1.0)}


def _latest(group: zarr.Group | None) -> zarr.Group | None:
    if group is None:
        return None
    keys = list(group.keys())
    return group[keys[-1]] if keys else None


def _text(array: np.ndarray) -> list[str]:
    return [bytes(row).split(b"\x00")[0].decode() for row in np.asarray(array)]


class RecordingData:
    """Everything the figures need, pulled once."""

    def __init__(self, zarr_path: Path):
        self.path = Path(zarr_path)
        self.root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        parent = self.root.get("analysis/chaser_distance_runs")
        if parent is None or not list(parent.keys()):
            raise ValueError(f"No chaser_distance_run in {zarr_path}")
        self.run = _latest(parent)
        self.recording_id = str(self.run.attrs.get("recording_id") or self.path.stem)
        self.ppm = float(self.run.attrs["pixels_per_mm_projector"])
        self.fps = float(self.run.attrs.get("fps", 100.0))
        self.geometry, _notes = resolve_arena_geometry(self.root, self.run, pixels_per_mm=self.ppm)

        self.fish = np.asarray(self.run["positions/fish_centroid_arena_xy"][:], dtype=np.float64) / self.ppm
        self.fish_valid = np.asarray(self.run["positions/fish_valid"][:], dtype=bool)
        es = self.run["epoch_summary"]
        self.epoch_labels = _text(es["label_bytes"][:])
        self.epoch_start = np.asarray(es["start_frame"][:], dtype=np.int64)
        self.epoch_end = np.asarray(es["end_frame"][:], dtype=np.int64)

        # object roles come from the CRA endpoint, never from index order
        self.roles: dict[int, str] = {}
        cra = _latest(self.run.get("cra_primary_endpoint"))
        if cra is not None:
            o = cra["objects"]
            idx = np.asarray(o["object_index"][:]).reshape(-1)
            code = np.asarray(o["object_role_code"][:]).reshape(-1)
            for i in range(len(idx)):
                self.roles[int(idx[i])] = "aggressive" if int(code[i]) == 1 else "inert"

        self.near_field = _latest(self.run.get("cra_near_field"))
        self.regimes = _latest(self.run.get("chaser_response_regimes"))
        self.bout = _latest(self.run.get("chaser_bout_response"))
        self.occupancy = _latest(self.run.get("chaser_radial_occupancy"))

    # ---- derived ----
    def wall_distance_mm(self, epoch: str) -> np.ndarray:
        if epoch not in self.epoch_labels:
            return np.asarray([])
        i = self.epoch_labels.index(epoch)
        s, e = int(self.epoch_start[i]), int(self.epoch_end[i]) + 1
        xy = self.fish[s:e]
        ok = self.fish_valid[s:e] & np.isfinite(xy).all(axis=1)
        if not np.any(ok):
            return np.asarray([])
        cx = float(self.geometry.center_x_px) / self.ppm
        cy = float(self.geometry.center_y_px) / self.ppm
        r = float(self.geometry.radius_px) / self.ppm
        return r - np.hypot(xy[ok, 0] - cx, xy[ok, 1] - cy)

    def dropout(self) -> dict[str, float]:
        out = {}
        for i, lab in enumerate(self.epoch_labels):
            s, e = int(self.epoch_start[i]), int(self.epoch_end[i]) + 1
            out[lab] = float(np.mean(~self.fish_valid[s:e]))
        return out

    def steering_bands(self) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
        """(bin centres, {(epoch, role): excess-by-band})"""
        if self.bout is None or "steering_excess_by_band" not in self.bout["object_vs_virtual"]:
            return np.asarray([]), {}
        c = self.bout
        centers = np.asarray(c["config/distance_bin_centers_mm"][:], dtype=np.float64)
        band = np.asarray(c["object_vs_virtual/steering_excess_by_band"][:], dtype=np.float64)
        eps = _text(c["epochs/label_bytes"][:])
        kinds = _text(c["references/kind_bytes"][:])
        ci = np.asarray(c["references/chaser_index"][:])
        objects = [i for i in range(len(kinds)) if kinds[i] == "object"]
        out: dict[tuple[str, str], np.ndarray] = {}
        for o_pos, r_idx in enumerate(objects):
            role = self.roles.get(int(ci[r_idx]), f"chaser{int(ci[r_idx])}")
            for e_idx, ep in enumerate(eps):
                out[(ep, role)] = band[e_idx, o_pos, :]
        return centers, out

    def freeze_curves(self) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
        if self.regimes is None:
            return np.asarray([]), {}
        c = self.regimes
        centers = np.asarray(c["config/distance_bin_centers_mm"][:], dtype=np.float64)
        imm = np.asarray(c["regimes/immobile_fraction"][:], dtype=np.float64)
        eps = _text(c["epochs/label_bytes"][:])
        ci = np.asarray(c["chasers/chaser_index"][:])
        out = {}
        for e_idx, ep in enumerate(eps):
            for k in range(len(ci)):
                out[(ep, self.roles.get(int(ci[k]), f"chaser{int(ci[k])}"))] = imm[e_idx, k, :]
        return centers, out

    def visit_counts(self) -> dict[tuple[str, str], int]:
        if self.bout is None:
            return {}
        c = self.bout
        eps = _text(c["epochs/label_bytes"][:])
        kinds = _text(c["references/kind_bytes"][:])
        ci = np.asarray(c["references/chaser_index"][:])
        nv = np.asarray(c["object_vs_virtual/near_visit_count"][:])
        out = {}
        for r_idx in range(len(kinds)):
            if kinds[r_idx] != "object":
                continue
            role = self.roles.get(int(ci[r_idx]), f"chaser{int(ci[r_idx])}")
            for e_idx, ep in enumerate(eps):
                out[(ep, role)] = int(nv[e_idx, r_idx])
        return out

    def thigmotaxis(self) -> tuple[float, float]:
        if self.near_field is None:
            return math.nan, math.nan
        s = self.near_field.attrs.get("summary", {})
        f = lambda k: float(s[k]) if s.get(k) is not None else math.nan  # noqa: E731
        return f("thigmotaxis_frac_pre"), f("thigmotaxis_frac_post")


# ======================================================================================
# per-recording sheet
# ======================================================================================


def render_recording_summary(zarr_path: Path, *, dpi: int = 130) -> bytes:
    d = RecordingData(zarr_path)
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.26)

    # --- 1. wall distance: the thigmotaxis result, as a distribution ---
    ax = fig.add_subplot(gs[0, 0])
    # Span the whole dish: a centre-dwelling fish sits ~40 mm from the wall, and clipping the
    # axis short piles it into a fake spike at the edge.
    r_arena = float(d.geometry.radius_px) / d.ppm
    bins = np.linspace(0, math.ceil(r_arena), 50)
    for ep, col in (("pre_event", "#64748b"), ("post_event", "#dc2626")):
        w = d.wall_distance_mm(ep)
        if w.size:
            ax.hist(w, bins=bins, density=True, histtype="step", lw=2,
                    color=col, label=f"{ep.replace('_event','')}  (median {np.median(w):.1f} mm)")
    ax.axvline(3.0, color="#94a3b8", ls=":", lw=1)
    ax.text(3.4, ax.get_ylim()[1] * 0.92, "touching\nthe wall", fontsize=7, color="#64748b")
    ax.axvline(r_arena, color="#94a3b8", ls=":", lw=1)
    ax.text(r_arena - 0.5, ax.get_ylim()[1] * 0.92, "dish\ncentre", fontsize=7,
            color="#64748b", ha="right")
    tp, tq = d.thigmotaxis()
    ax.set_title(f"Distance from the wall\nthigmotaxis {tp:.2f} → {tq:.2f}", fontsize=10)
    ax.set_xlabel("distance to wall (mm)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # --- 2. steering excess dose-response: THE result, with its control baked in ---
    ax = fig.add_subplot(gs[0, 1])
    centers, bands = d.steering_bands()
    if centers.size:
        for (ep, role), v in bands.items():
            if ep not in EPOCH_STYLE or role not in OBJ_COLOR:
                continue
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            ax.plot(centers[ok], v[ok], marker="o", ms=4, lw=1.8, color=OBJ_COLOR[role],
                    label=f"{role} · {ep.replace('_event','')}", **EPOCH_STYLE[ep])
    ax.axhline(0, color="#334155", ls="-", lw=1)
    ax.axvspan(8, 16, color="#fbbf24", alpha=0.12, lw=0)
    ax.text(12, ax.get_ylim()[1] * 0.9, "shell", fontsize=7, ha="center", color="#b45309")
    ax.set_title("Steering excess vs wall-matched virtual twins\n(+ = bout re-aims fish to pass WIDER)", fontsize=10)
    ax.set_xlabel("distance to object at bout onset (mm)")
    ax.set_ylabel("Δ predicted miss, object − virtual (mm)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # --- 3. freeze curve ---
    ax = fig.add_subplot(gs[0, 2])
    fc_centers, curves = d.freeze_curves()
    if fc_centers.size:
        for (ep, role), v in curves.items():
            if role != "aggressive" or ep not in ("pre_event", "training_event", "post_event"):
                continue
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            col = {"pre_event": "#64748b", "training_event": "#dc2626", "post_event": "#2563eb"}[ep]
            ax.plot(fc_centers[ok], v[ok], marker="o", ms=3.5, lw=1.6, color=col,
                    label=ep.replace("_event", ""))
    ax.set_ylim(0, 1)
    ax.set_title("Freeze curve — P(frozen | distance)\naggressive object", fontsize=10)
    ax.set_xlabel("distance to object (mm)")
    ax.set_ylabel("P(speed < 1 mm/s)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # --- 4. where the fish actually was ---
    ax = fig.add_subplot(gs[1, 0])
    cx = float(d.geometry.center_x_px) / d.ppm
    cy = float(d.geometry.center_y_px) / d.ppm
    rr = float(d.geometry.radius_px) / d.ppm
    for ep, col in (("pre_event", "#64748b"), ("post_event", "#dc2626")):
        if ep not in d.epoch_labels:
            continue
        i = d.epoch_labels.index(ep)
        s, e = int(d.epoch_start[i]), int(d.epoch_end[i]) + 1
        xy = d.fish[s:e][d.fish_valid[s:e]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.size:
            ax.plot(xy[::20, 0], xy[::20, 1], ".", ms=0.6, alpha=0.25, color=col,
                    label=ep.replace("_event", ""))
    ax.add_patch(plt.Circle((cx, cy), rr, fill=False, color="#334155", lw=1.6))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Fish occupancy\ngeometry: {d.geometry.status}", fontsize=10)
    lg = ax.legend(fontsize=7, markerscale=8)
    for h in lg.legend_handles:
        h.set_alpha(1.0)

    # --- 5. tracking QC — the missing data, on the sheet, not in a footnote ---
    ax = fig.add_subplot(gs[1, 1])
    dr = d.dropout()
    labs = [e for e in ("pre_event", "training_event", "post_event") if e in dr]
    vals = [100 * dr[e] for e in labs]
    cols = ["#dc2626" if v > 10 else "#16a34a" for v in vals]
    ax.bar(range(len(labs)), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9,
                fontweight="bold" if v > 10 else "normal")
    ax.axhline(10, color="#dc2626", ls="--", lw=0.9, alpha=0.7)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([l.replace("_event", "") for l in labs], fontsize=8)
    ax.set_ylabel("fish tracking dropout (%)")
    ax.set_ylim(0, max(105, max(vals + [10]) * 1.25))
    ax.set_title("Tracking dropout\nrecall collapses when the fish freezes →\nfreeze metrics are LOWER BOUNDS", fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    # --- 6. the effective n ---
    ax = fig.add_subplot(gs[1, 2])
    vc = d.visit_counts()
    rows = [(ep, role) for ep in ("pre_event", "post_event") for role in ("aggressive", "inert")
            if (ep, role) in vc]
    if rows:
        y = np.arange(len(rows))
        ax.barh(y, [vc[k] for k in rows],
                color=[OBJ_COLOR[r] for _e, r in rows],
                alpha=[0.6 if e == "pre_event" else 1.0 for e, _r in rows][0], height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{e.replace('_event','')} · {r}" for e, r in rows], fontsize=8)
        for i, k in enumerate(rows):
            ax.text(vc[k] + 0.15, i, str(vc[k]), va="center", fontsize=9)
        ax.axvline(10, color="#dc2626", ls="--", lw=0.9)
        ax.set_xlabel("independent visits within 15 mm")
        ax.set_title("Effective sample size\n(bouts inside one visit are NOT independent)", fontsize=9)
    ax.grid(alpha=0.2, axis="x")

    fig.suptitle(f"{d.recording_id}", fontsize=13, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ======================================================================================
# cohort figures
# ======================================================================================


def _cohort_records(registry: Path, pattern: str) -> list[Path]:
    conn = sqlite3.connect(str(registry))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT DISTINCT recording_id, zarr_path FROM dataset_context_current
           WHERE recording_id LIKE ? AND zarr_use='analysis' AND dataset_status='active'
           ORDER BY recording_id""", (pattern,)).fetchall()
    seen, out = set(), []
    for r in rows:                     # the registry can hold two paths for one recording_id
        if r["recording_id"] in seen:
            continue
        seen.add(r["recording_id"])
        p = Path(r["zarr_path"])
        if p.exists():
            out.append(p)
    return out


def render_cohort_figures(registry: Path, pattern: str = "%GoodCopBadCop%") -> tuple[bytes, list[RecordingData]]:
    data: list[RecordingData] = []
    for p in _cohort_records(registry, pattern):
        try:
            data.append(RecordingData(p))
        except Exception:
            continue
    if not data:
        raise ValueError("No usable recordings.")

    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    # --- A. thigmotaxis, every fish drawn ---
    ax = fig.add_subplot(gs[0, 0])
    pre, post = [], []
    for d in data:
        a, b = d.thigmotaxis()
        if math.isfinite(a) and math.isfinite(b):
            pre.append(a); post.append(b)
            ax.plot([0, 1], [a, b], "-", lw=0.9,
                    color="#dc2626" if b > a else "#2563eb", alpha=0.5)
            ax.plot([0, 1], [a, b], ".", ms=4, color="#334155", alpha=0.6)
    pre, post = np.array(pre), np.array(post)
    ax.plot([0, 1], [pre.mean(), post.mean()], "-", lw=3.5, color="#111827", zorder=5)
    ax.plot([0, 1], [pre.mean(), post.mean()], "o", ms=8, color="#111827", zorder=6)
    ax.set_xlim(-0.25, 1.25); ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
    ax.set_ylabel("thigmotaxis fraction")
    n_up = int((post > pre).sum())
    ax.set_title(f"Thigmotaxis, every fish (n={len(pre)})\n"
                 f"{pre.mean():.3f} → {post.mean():.3f}   ({n_up} up, {len(pre)-n_up} down)",
                 fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    # --- B. the dose-response, cohort mean ± SEM, with the inert control ---
    ax = fig.add_subplot(gs[0, 1:])
    centers = None
    stack: dict[tuple[str, str], list[np.ndarray]] = {}
    for d in data:
        c, bands = d.steering_bands()
        if not c.size:
            continue
        centers = c
        for k, v in bands.items():
            stack.setdefault(k, []).append(v)
    if centers is not None:
        for (ep, role), vs in sorted(stack.items()):
            if ep not in EPOCH_STYLE or role not in OBJ_COLOR:
                continue
            A = np.vstack(vs)
            n = np.sum(np.isfinite(A), axis=0)
            mean = np.nanmean(A, axis=0)
            sem = np.nanstd(A, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            ok = n >= 8
            ax.plot(centers[ok], mean[ok], marker="o", ms=5, lw=2.2, color=OBJ_COLOR[role],
                    label=f"{role} · {ep.replace('_event','')} (n≥8)", **EPOCH_STYLE[ep])
            ax.fill_between(centers[ok], (mean - sem)[ok], (mean + sem)[ok],
                            color=OBJ_COLOR[role], alpha=0.13, lw=0)
    ax.axhline(0, color="#334155", lw=1)
    ax.axvspan(8, 16, color="#fbbf24", alpha=0.13, lw=0)
    ax.set_xlim(0, 32)   # beyond ~32 mm the bins are wide, sparse and noisy; the decay is the point
    ax.text(12, ax.get_ylim()[1] * 0.92, "8–16 mm shell", ha="center", fontsize=8, color="#b45309")
    ax.set_title("Steering excess over wall-matched virtual controls — cohort mean ± SEM\n"
                 "a real object effect is LOCALIZED and decays; an artifact is flat", fontsize=10)
    ax.set_xlabel("distance to object at bout onset (mm)")
    ax.set_ylabel("Δ predicted miss, object − virtual (mm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # --- C. the shell, paired, per fish ---
    ax = fig.add_subplot(gs[1, 0])
    shell = {}
    for role in ("aggressive", "inert"):
        a_, b_ = [], []
        for d in data:
            c, bands = d.steering_bands()
            if not c.size:
                continue
            m = (c >= 8) & (c <= 16)
            va = bands.get(("pre_event", role)); vb = bands.get(("post_event", role))
            if va is None or vb is None:
                continue
            x = np.nanmean(va[m]); y = np.nanmean(vb[m])
            if math.isfinite(x) and math.isfinite(y):
                a_.append(x); b_.append(y)
        shell[role] = (np.array(a_), np.array(b_))
    for j, role in enumerate(("aggressive", "inert")):
        a_, b_ = shell[role]
        if not a_.size:
            continue
        x0, x1 = j * 2, j * 2 + 1
        for u, v in zip(a_, b_):
            ax.plot([x0, x1], [u, v], "-", color=OBJ_COLOR[role], alpha=0.28, lw=0.9)
        ax.plot([x0, x1], [a_.mean(), b_.mean()], "-o", lw=3, ms=7, color=OBJ_COLOR[role], zorder=5)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["pre", "post", "pre", "post"], fontsize=8)
    lo = ax.get_ylim()[0]
    span = ax.get_ylim()[1] - lo
    ax.text(0.5, lo - 0.13 * span, "AGGRESSIVE", ha="center", fontsize=9,
            color=OBJ_COLOR["aggressive"], fontweight="bold", clip_on=False)
    ax.text(2.5, lo - 0.13 * span, "inert (control)", ha="center", fontsize=9,
            color=OBJ_COLOR["inert"], clip_on=False)
    ax.set_ylabel("steering excess in the 8–16 mm shell (mm)")
    ax.set_title("The shell, per fish\nthe inert object is the control: it should not move", fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    # --- D. detector dropout, the bias that matters ---
    ax = fig.add_subplot(gs[1, 1])
    tr = np.array([d.dropout().get("training_event", np.nan) for d in data]) * 100
    tr = tr[np.isfinite(tr)]
    order = np.argsort(tr)
    cols = ["#dc2626" if v > 10 else "#94a3b8" for v in tr[order]]
    ax.bar(range(len(tr)), tr[order], color=cols, width=0.85)
    ax.axhline(10, color="#dc2626", ls="--", lw=1)
    ax.set_xlabel("recording (sorted)")
    ax.set_ylabel("training-epoch dropout (%)")
    ax.set_title(f"Detector dropout in the chase\n{int((tr>10).sum())}/{len(tr)} recordings >10% — "
                 f"recall fails when the fish freezes", fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    # --- E. the effective n, cohort-wide ---
    ax = fig.add_subplot(gs[1, 2])
    for j, role in enumerate(("aggressive", "inert")):
        v = np.array([d.visit_counts().get(("post_event", role), np.nan) for d in data], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=np.arange(-0.5, max(16, v.max() + 2)), alpha=0.6,
                    color=OBJ_COLOR[role], label=f"{role} (median {np.median(v):.0f})")
    ax.axvline(10, color="#dc2626", ls="--", lw=1)
    ax.set_xlabel("independent visits within 15 mm (post)")
    ax.set_ylabel("recordings")
    ax.set_title("Effective n per fish\nvisits, not bouts — most fish are below 10", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(f"GoodCopBadCop cohort — {len(data)} recordings", fontsize=13, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), data


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--recording-like", default="%GoodCopBadCop%")
    p.add_argument("--zarr", type=Path, default=None, help="Render a single recording sheet.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-recording", action="store_true", help="Also write one sheet per recording.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.zarr:
        png = render_recording_summary(args.zarr)
        out = args.out_dir / f"{RecordingData(args.zarr).recording_id}_summary.png"
        out.write_bytes(png)
        print(f"wrote {out}")
        return 0

    from fisheye.registry.db import RegistryPaths
    registry = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    png, data = render_cohort_figures(registry, str(args.recording_like))
    out = args.out_dir / "cohort_summary.png"
    out.write_bytes(png)
    print(f"wrote {out}   ({len(data)} recordings)")

    if args.per_recording:
        for d in data:
            try:
                p = args.out_dir / f"{d.recording_id}_summary.png"
                p.write_bytes(render_recording_summary(d.path))
                print(f"  wrote {p.name}")
            except Exception as exc:
                print(f"  FAILED {d.recording_id}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
