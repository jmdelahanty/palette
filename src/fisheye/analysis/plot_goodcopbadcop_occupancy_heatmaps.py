#!/usr/bin/env python
"""Reproducible: pre-vs-post occupancy heatmaps for the top learned-spatial-avoidance fish.

Ranks fish by `occ_learned` (post-minus-pre near-object dwell vs a wall-matched virtual,
from analyze_goodcopbadcop_per_fish) and plots each top fish's arena occupancy heatmap in
pre vs post, with the aggressive object's position and its 15 mm near-zone marked. This is
the visual confirmation of the individual "post heatmap stays off the chaser's zone" fish
that the cohort mean (~null) hides.

Run (palette env):
    python -m fisheye.analysis.plot_goodcopbadcop_occupancy_heatmaps            # top 6
    python -m fisheye.analysis.plot_goodcopbadcop_occupancy_heatmaps --top 8
    python -m fisheye.analysis.plot_goodcopbadcop_occupancy_heatmaps --recording-id 22-33-50Z_arena_1

Positions are arena-frame, converted to mm relative to the arena centre. Reads the
canonical registry; writes the figure OUTSIDE the repo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fisheye.analysis.goodcopbadcop_common import (
    figures_dir,
    latest,
    load_epochs,
    nav,
    resolve_cohort,
    resolve_object_roles,
)
from fisheye.analysis.analyze_goodcopbadcop_per_fish import spatial_avoidance
from fisheye.shared.arena_geometry import resolve_arena_geometry

NEAR_MM = 15.0
BINS = 44


def load_positions(zp: str):
    r = zarr.open_group(zp, mode="r")
    cd = latest(nav(r, ["analysis", "chaser_distance_runs"]))
    roles = resolve_object_roles(r)
    agg = roles["aggressive"]
    fish = np.asarray(cd["positions"]["fish_centroid_arena_xy"][:], float)
    fvalid = np.asarray(cd["positions"]["fish_valid"][:], bool)
    chas = np.asarray(cd["positions"]["chaser_arena_xy"][:], float)
    cvalid = np.asarray(cd["positions"]["chaser_valid"][:], bool)[:, agg]
    ppm = float(cd.attrs.get("pixels_per_mm_projector"))
    geo, _ = resolve_arena_geometry(r, cd, pixels_per_mm=ppm)
    cx, cy = geo.center_x_px, geo.center_y_px
    if cx is None:
        raise ValueError("no arena geometry")
    fx = (fish[:, 0] - cx) / ppm
    fy = (fish[:, 1] - cy) / ppm
    ox = (chas[:, agg, 0] - cx) / ppm
    oy = (chas[:, agg, 1] - cy) / ppm
    return {"fx": fx, "fy": fy, "fvalid": fvalid, "ox": ox, "oy": oy, "cvalid": cvalid,
            "radius_mm": geo.radius_px / ppm, "epochs": load_epochs(r)}


def epoch_slice(valid, rng):
    m = valid.copy()
    m[:rng[0]] = False
    m[rng[1] + 1:] = False
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=6, help="Number of highest-occ_learned fish.")
    ap.add_argument("--recording-id", default=None, help="Plot one recording (substring) instead of the top-N.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-18")
    args = ap.parse_args()
    out_dir = args.out_dir or figures_dir()

    cohort = resolve_cohort()
    if args.recording_id:
        selected = [(rid, zp, np.nan) for rid, zp in cohort if args.recording_id in rid]
    else:
        scored = []
        for rid, zp in cohort:
            try:
                ol = spatial_avoidance(zp).get("occ_learned", np.nan)
            except Exception:
                ol = np.nan
            if np.isfinite(ol):
                scored.append((rid, zp, ol))
        scored.sort(key=lambda t: t[2], reverse=True)
        selected = scored[:args.top]
    n = len(selected)
    if n == 0:
        print("no recordings selected")
        return

    fig, axes = plt.subplots(n, 2, figsize=(7.0, 3.3 * n), squeeze=False)
    for ri, (rid, zp, ol) in enumerate(selected):
        try:
            d = load_positions(zp)
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        R = d["radius_mm"]
        ext = [-R * 1.05, R * 1.05, -R * 1.05, R * 1.05]
        edges = np.linspace(-R * 1.05, R * 1.05, BINS + 1)
        for ci, ep in enumerate(("pre", "post")):
            ax = axes[ri, ci]
            if ep not in d["epochs"]:
                ax.set_axis_off()
                continue
            rng = d["epochs"][ep]
            m = epoch_slice(d["fvalid"], rng) & np.isfinite(d["fx"])
            h, _, _ = np.histogram2d(d["fx"][m], d["fy"][m], bins=[edges, edges], density=True)
            ax.imshow(h.T, origin="lower", extent=ext, cmap="magma", aspect="equal",
                      vmax=np.percentile(h[h > 0], 98) if (h > 0).any() else None)
            ax.add_patch(Circle((0, 0), R, fill=False, ec="#88ccff", lw=1.2, alpha=0.7))
            # aggressive object position this epoch (median over valid frames)
            om = epoch_slice(d["cvalid"] & d["fvalid"], rng) & np.isfinite(d["ox"])
            if om.sum() > 50:
                ox, oy = np.median(d["ox"][om]), np.median(d["oy"][om])
                ax.add_patch(Circle((ox, oy), NEAR_MM, fill=False, ec="#39ff14", lw=1.6, ls="--"))
                ax.scatter([ox], [oy], marker="*", s=180, color="#39ff14", edgecolor="k", zorder=5,
                           label="aggressive object")
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])
            title = ep.upper()
            if ci == 0:
                title = f"{rid.split('_GoodCop')[0]}\nocc_learned={ol:+.3f}   {ep.upper()}"
            ax.set_title(title, fontsize=9, weight="bold" if ci == 0 else "normal")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper right", labelcolor="#333")
    fig.suptitle("Fish occupancy pre vs post — top learned-spatial-avoidance fish\n"
                 "(green star = aggressive object; dashed = 15 mm near-zone it avoids)",
                 fontsize=12, weight="bold", y=1.005)
    fig.tight_layout()
    tag = args.recording_id or f"top{args.top}"
    out = out_dir / f"goodcopbadcop_occupancy_heatmaps_{tag}_{args.tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}  (n={n})")
    for rid, zp, ol in selected:
        print(f"  {rid.split('_GoodCop')[0]:30s} occ_learned={ol:+.3f}")


if __name__ == "__main__":
    main()
