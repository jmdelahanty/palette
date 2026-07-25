#!/usr/bin/env python
"""Reproducible meeting figures: GoodCopBadCop mid-band freeze/immobility avoidance readout.

Renders the three result figures from the distance-resolved immobility curve
(chaser_response_regimes `immobile_fraction`), computed across the reachable
GoodCopBadCop analysis zarrs (resolved from the registry, so path-move safe):

  1. goodcopbadcop_freeze_curve_<tag>.png    P(immobile) vs distance, pre/post, red/inert
  2. goodcopbadcop_freeze_paired_<tag>.png    paired 7-18mm band pre->post per recording
  3. goodcopbadcop_freeze_summary_<tag>.png   the three contrasts (red mid, inert mid, red far)

RETRACTED (2026-07-17): the mid-band immobility "avoidance" effect this script was written to
show (raw Δ+0.177, p=0.002) was a RAW-TRACKING-NOISE ARTIFACT. chaser_response_regimes now
classifies immobility on speed_smoothed_mm, so this script draws the NULL (aggressive mid-band
Δ~+0.004, p~0.85; inert and far also ~0). The definitive figure is the raw-vs-smoothed contrast
in scratch/freeze_corrected_plots (curve/summary PNGs). Keep this script only for the null; do
NOT present its output as an avoidance result. What survived the clean-signal check: the escape
response (12x during chase, 12/12, p=0.0005). See docs/archive/goodcopbadcop_avoidance_readout_survey.md
item 1 and the chaser_response_regimes contract.

Run (palette env):
    ~/miniconda3/envs/palette-py311/bin/python -m fisheye.analysis.plot_goodcopbadcop_freeze

Figures are written OUTSIDE the repo (to $PALETTE_RECORDINGS_ROOT/figures); this script is
committed but its output is not.

Options:
    --out-dir  where to write (default: $PALETTE_RECORDINGS_ROOT/figures).
    --tag      filename suffix (default: 2026-07-17, reproduces the meeting artifacts).

Provenance: /nvme1/recordings/figures/goodcopbadcop_freeze_figures_2026-07-17_PROVENANCE.txt
Method doc: docs/archive/goodcopbadcop_avoidance_readout_survey.md
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fisheye.analysis.chaser_response_regimes import build_chaser_response_regimes_result as build
from fisheye.analysis.cra_primary_endpoint import resolve_object_roles_from_protocol_payload
from fisheye.analysis.goodcopbadcop_common import resolve_cohort as cohort
from fisheye.analysis.goodcopbadcop_common import role_index, role_name
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value, paired_sign_flip_p_value

# Figures live OUTSIDE the repo (this script is committed; its output is not).
FIGURES_DIR = Path(os.environ.get("PALETTE_RECORDINGS_ROOT", "/nvme1/recordings")) / "figures"
BAND = (7.0, 18.0); FAR = (25.0, 50.0); MIN_BIN = 20; MIN_BAND_FRAMES = 150
AGG_C = "#c1435b"; INERT_C = "#3a7ca5"; GRID = "#e6e6e6"
# cohort() is the shared, registry-resolved, duplicate-deduped resolver (see goodcopbadcop_common).


def roles(res):
    """Object index AND rendered colour per role, from the experiment metadata."""
    r = zarr.open_group(res.zarr_path, mode="r"); stim = r[res.source_stimulus_path]
    by = {role_name(o): o for o in resolve_object_roles_from_protocol_payload(
        json.loads(str(stim.attrs["protocol_json"])))}
    a, i = by["aggressive"], by["inert"]
    return role_index(a), role_index(i), a.raw_color_hex, i.raw_color_hex


def emap(res):
    return {e.label.split("_")[0]: i for i, e in enumerate(res.epochs)}


def band_mean(res, ei, ch, lo, hi):
    ctr = res.distance_bin_centers_mm; sel = (ctr >= lo) & (ctr <= hi)
    fc = res.frame_count[ei, ch, sel]; im = res.immobile_fraction[ei, ch, sel]
    ok = np.isfinite(im) & (fc > 0)
    return float(np.average(im[ok], weights=fc[ok])) if fc[ok].sum() >= MIN_BAND_FRAMES else np.nan


def paired(pre, post):
    mask = np.isfinite(pre) & np.isfinite(post); d = post[mask] - pre[mask]
    wp = wilcoxon_signed_rank_p_value(d)[0]
    sfp = paired_sign_flip_p_value(d, iterations=20000, rng=np.random.default_rng(0))[0]
    return d, wp, sfp, int(mask.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--tag", default="2026-07-17")
    args = ap.parse_args(argv)
    OUT = args.out_dir; OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "figure.dpi": 140, "savefig.dpi": 160, "font.family": "DejaVu Sans"})

    recs = cohort()
    centers = None
    curves = {("agg", "pre"): [], ("agg", "post"): [], ("inert", "pre"): [], ("inert", "post"): []}
    scal = {k: [] for k in ("agg_pre", "agg_post", "in_pre", "in_post", "aggfar_pre", "aggfar_post")}
    labels = []
    agg_hexes, inert_hexes = set(), set()
    for rid, zp in recs:
        try:
            res = build(Path(zp)); a, inr, ahex, ihex = roles(res); em = emap(res)
            ip, io = em["pre"], em["post"]
        except Exception as exc:  # noqa: BLE001
            print("skip", rid, exc); continue
        agg_hexes.add(ahex); inert_hexes.add(ihex)
        if centers is None:
            centers = res.distance_bin_centers_mm
        for role, ch in (("agg", a), ("inert", inr)):
            for ep, ei in (("pre", ip), ("post", io)):
                im = res.immobile_fraction[ei, ch].copy(); fc = res.frame_count[ei, ch]
                im[fc < MIN_BIN] = np.nan
                curves[(role, ep)].append(im)
        scal["agg_pre"].append(band_mean(res, ip, a, *BAND)); scal["agg_post"].append(band_mean(res, io, a, *BAND))
        scal["in_pre"].append(band_mean(res, ip, inr, *BAND)); scal["in_post"].append(band_mean(res, io, inr, *BAND))
        scal["aggfar_pre"].append(band_mean(res, ip, a, *FAR)); scal["aggfar_post"].append(band_mean(res, io, a, *FAR))
        labels.append(rid.split("_GoodCop")[0])
    for k in scal:
        scal[k] = np.array(scal[k], float)
    n_rec = len(labels)
    if n_rec == 0:
        raise SystemExit("No reachable GoodCopBadCop analysis zarrs.")

    # Series colours come from the experiment metadata; use them if consistent across the
    # cohort, else fall back to the palette. Labels are always by ROLE (counterbalance-proof).
    agg_color = next(iter(agg_hexes)) if len(agg_hexes) == 1 else AGG_C
    inert_color = next(iter(inert_hexes)) if len(inert_hexes) == 1 else INERT_C
    if len(agg_hexes) == 1 and len(inert_hexes) == 1:
        color_note = (f"aggressive = {agg_color} · inert = {inert_color}  "
                      f"(fixed across all {n_rec} recordings — colour confounded with role)")
    else:
        color_note = "object colours vary across recordings; series labelled by role (from metadata)"

    def curve_stats(role, ep):
        arr = np.array(curves[(role, ep)], float)
        m = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(np.isfinite(arr), axis=0))
        return m, sem

    # FIG 1 -- freeze curve
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axvspan(BAND[0], BAND[1], color="#f2c94c", alpha=0.16, zorder=0, label="_")
    styles = {("agg", "post"): (agg_color, "-", 2.4, "o"), ("agg", "pre"): (agg_color, "--", 1.6, "o"),
              ("inert", "post"): (inert_color, "-", 2.4, "s"), ("inert", "pre"): (inert_color, "--", 1.6, "s")}
    namemap = {("agg", "pre"): "Aggressive · pre", ("agg", "post"): "Aggressive · post",
               ("inert", "pre"): "Inert · pre", ("inert", "post"): "Inert · post"}
    for key in [("inert", "pre"), ("inert", "post"), ("agg", "pre"), ("agg", "post")]:
        m, sem = curve_stats(*key); c, ls, lw, mk = styles[key]
        ok = np.isfinite(m)
        ax.plot(centers[ok], m[ok], ls, color=c, lw=lw, marker=mk, ms=5,
                markerfacecolor=c if key[1] == "post" else "white", markeredgecolor=c, label=namemap[key], zorder=3)
        ax.fill_between(centers[ok], (m - sem)[ok], (m + sem)[ok], color=c, alpha=0.12, zorder=1)
    ax.set_xlim(3, 52); ax.set_ylim(0, 0.8)
    ax.set_xlabel("Distance to object (mm)"); ax.set_ylabel("P(fish immobile)")
    ax.set_title("No mid-band immobility difference on the smoothed signal (pre vs post)",
                 fontsize=13, weight="bold", pad=12)
    ax.text(12.5, 0.045, "avoidance shell\n7–18 mm", ha="center", va="bottom", fontsize=9, color="#8a6d00")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.text(0.5, -0.16, f"n = {n_rec} recordings (3 sessions, 2026-06-14). Bands = ±SEM across recordings. "
            "Immobility = centroid speed < 1 mm/s (provisional; see caveats).",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    ax.text(0.5, -0.205, color_note, transform=ax.transAxes, ha="center", fontsize=7.5, color="#888")
    fig.tight_layout(); fig.savefig(OUT / f"goodcopbadcop_freeze_curve_{args.tag}.png", bbox_inches="tight")
    plt.close(fig)

    # FIG 2 -- paired slopegraph
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.4), sharey=True)
    for ax, (pk, pok, col, ttl) in zip(axes, [("agg_pre", "agg_post", agg_color, "Aggressive object"),
                                              ("in_pre", "in_post", inert_color, "Inert object")]):
        pre, post = scal[pk], scal[pok]
        d, wp, sfp, nn = paired(pre, post)
        for i in range(n_rec):
            if np.isfinite(pre[i]) and np.isfinite(post[i]):
                up = post[i] > pre[i]
                ax.plot([0, 1], [pre[i], post[i]], "-", color=col if up else "#bbb", lw=1.4, alpha=0.75, zorder=2)
                ax.scatter([0, 1], [pre[i], post[i]], color=col, s=22, zorder=3)
        mpre, mpost = np.nanmean(pre), np.nanmean(post)
        ax.plot([0, 1], [mpre, mpost], "-", color="black", lw=3, zorder=4)
        ax.scatter([0, 1], [mpre, mpost], color="black", s=55, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"]); ax.set_xlim(-0.25, 1.25)
        ax.set_title(f"{ttl}\nΔ={np.nanmean(d):+.3f}  ({int((d>0).sum())}/{nn} up)  Wilcoxon p={wp:.3f}", fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("P(fish immobile), 7–18 mm band"); axes[0].set_ylim(0.15, 0.8)
    fig.suptitle("No post-training immobility change near either object (smoothed signal)",
                 fontsize=13, weight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / f"goodcopbadcop_freeze_paired_{args.tag}.png", bbox_inches="tight")
    plt.close(fig)

    # FIG 3 -- summary deltas
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    rows = [("Aggressive,\n7–18 mm", "agg_pre", "agg_post", agg_color),
            ("Inert,\n7–18 mm", "in_pre", "in_post", inert_color),
            ("Aggressive,\n25–50 mm (far ctrl)", "aggfar_pre", "aggfar_post", "#8a8a8a")]
    ys = np.arange(len(rows))[::-1]
    summary = []
    for y, (name, pk, pok, col) in zip(ys, rows):
        d, wp, sfp, nn = paired(scal[pk], scal[pok])
        ax.scatter(d, np.full_like(d, y, dtype=float) + np.random.default_rng(1).uniform(-0.08, 0.08, len(d)),
                   color=col, s=26, alpha=0.6, zorder=2)
        md = np.nanmean(d); se = np.nanstd(d) / np.sqrt(len(d))
        ax.errorbar(md, y, xerr=se, fmt="o", color="black", ms=9, capsize=4, lw=2, zorder=4)
        ax.text(0.63, y, f"Δ={md:+.3f}\np={wp:.3f}\nn={nn}", va="center", fontsize=9.5)
        summary.append((name.replace("\n", " "), md, wp, nn))
    ax.axvline(0, color="#999", lw=1, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.25, 0.85); ax.set_xlabel("Δ P(immobile), post − pre")
    ax.set_title("No immobility change on the smoothed signal (any object or band)", fontsize=13, weight="bold", pad=10)
    ax.text(0.5, -0.17, "Each dot = one recording (post − pre Δ); black = mean ± SEM across recordings.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / f"goodcopbadcop_freeze_summary_{args.tag}.png", bbox_inches="tight")
    plt.close(fig)

    print(f"n_rec = {n_rec}   tag = {args.tag}   out = {OUT}")
    for name, md, wp, nn in summary:
        print(f"  {name:34s} Δ={md:+.3f}  p={wp:.3f}  n={nn}")
    for stem in ("curve", "paired", "summary"):
        print(f"  wrote goodcopbadcop_freeze_{stem}_{args.tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
