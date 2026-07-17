#!/usr/bin/env python
"""Reproducible: is the near-object bout-vigor gradient learned or innate? (pre vs post)

The near-object bout-vigor gradient = median bout peak speed near (7-18 mm) minus far
(25-50 mm), per object per epoch. If learning drove it, the AGGRESSIVE near-far gradient
would grow from pre to post. This script splits the gradient by epoch and tests growth.

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_bout_vigor_prepost

Key result (full canonical cohort, n=29): the near-object bout-vigor gradient WEAKENS to
non-significance -- aggressive pre gradient ~+1.05 mm/s, p~0.14 vs 0 (was +1.54, p~0.042
on the n=12 slice); learned-vs-innate Δ p~0.11. The gradient does NOT grow with training
(so still not "learned avoidance"), but the innate red-vigor claim is marginal at full n
-- the n=12 slice was optimistic. Report this as weak/suggestive, not established.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fisheye.analysis.goodcopbadcop_common import (
    figures_dir,
    load_epochs,
    open_distance_run,
    resolve_cohort,
    resolve_object_roles,
)
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

EDGES = np.array([3, 6, 9, 13, 18, 25, 35, 50], float)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
NEAR = (7, 18)
FAR = (25, 50)
MIN_BOUTS = 15
AGG_C = "#c1435b"
INERT_C = "#3a7ca5"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "font.family": "DejaVu Sans",
                     "savefig.dpi": 160})


def load(zp: str):
    r = zarr.open_group(zp, mode="r")
    cd = open_distance_run(r)
    dist = np.asarray(cd["distances"]["distance_mm"][:], float)
    b = cd["chaser_bout_response"][sorted(cd["chaser_bout_response"].group_keys())[-1]]["bouts"]
    bv = np.asarray(b["valid"][:], bool)
    bs = np.asarray(b["start_frame"][:], np.int64)[bv]
    pk = np.asarray(b["peak_speed_mm_s"][:], float)[bv]
    return dist, bs, pk, resolve_object_roles(r), load_epochs(r)


def band_median(dist, bs, pk, idx, rng, band) -> float:
    s, e = rng
    in_epoch = (bs >= s) & (bs <= e)
    d = dist[bs, idx]
    m = in_epoch & np.isfinite(d) & (d >= band[0]) & (d <= band[1]) & np.isfinite(pk)
    return np.median(pk[m]) if m.sum() >= MIN_BOUTS else np.nan


def paired(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    d = b[m] - a[m]
    return (np.mean(d), wilcoxon_signed_rank_p_value(d)[0], int(m.sum())) if m.sum() >= 3 else (np.nan, np.nan, int(m.sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-17")
    args = ap.parse_args()
    out_dir = args.out_dir or figures_dir()

    grad = {obj: {ep: [] for ep in ("pre", "post")} for obj in ("agg", "inert")}
    pool = {(obj, ep): {"d": [], "v": []} for obj in ("agg", "inert") for ep in ("pre", "post")}
    n = 0
    for rid, zp in resolve_cohort():
        try:
            dist, bs, pk, roles, epochs = load(zp)
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        n += 1
        for obj, idx in (("agg", roles["aggressive"]), ("inert", roles["inert"])):
            for ep in ("pre", "post"):
                if ep not in epochs:
                    grad[obj][ep].append(np.nan)
                    continue
                near = band_median(dist, bs, pk, idx, epochs[ep], NEAR)
                far = band_median(dist, bs, pk, idx, epochs[ep], FAR)
                grad[obj][ep].append(near - far)
                s, e = epochs[ep]
                in_epoch = (bs >= s) & (bs <= e)
                d = dist[bs, idx]
                ok = in_epoch & np.isfinite(d) & np.isfinite(pk)
                pool[(obj, ep)]["d"].append(d[ok])
                pool[(obj, ep)]["v"].append(pk[ok])
    for obj in grad:
        for ep in grad[obj]:
            grad[obj][ep] = np.array(grad[obj][ep], float)

    def curve(obj, ep):
        d = np.concatenate(pool[(obj, ep)]["d"])
        v = np.concatenate(pool[(obj, ep)]["v"])
        med = np.full(CTR.size, np.nan)
        for i in range(CTR.size):
            m = (d >= EDGES[i]) & (d < EDGES[i + 1])
            if m.sum() >= MIN_BOUTS:
                med[i] = np.median(v[m])
        return med

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    styles = {("agg", "pre"): (AGG_C, "--", "o"), ("agg", "post"): (AGG_C, "-", "o"),
              ("inert", "pre"): (INERT_C, "--", "s"), ("inert", "post"): (INERT_C, "-", "s")}
    for (obj, ep), (col, ls, mk) in styles.items():
        med = curve(obj, ep)
        ok = np.isfinite(med)
        ax.plot(CTR[ok], med[ok], ls, color=col, lw=2.2 if ep == "post" else 1.5, marker=mk, ms=5,
                markerfacecolor=col if ep == "post" else "white", markeredgecolor=col,
                label=f"{obj} · {ep}")
    ax.axvspan(*NEAR, color="#f2c94c", alpha=0.12, zorder=0)
    ax.set_xlabel("onset distance to object (mm)")
    ax.set_ylabel("bout peak speed (mm/s)")
    ax.set_xlim(3, 52)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Near-object bout vigor, pre vs post -- is the gradient learned?", fontsize=12.5, weight="bold")

    print("Gradient = median peak(7-18mm) - peak(25-50mm), per recording. Positive = faster bouts near.\n")
    for obj in ("agg", "inert"):
        gp, pp, nn = paired(grad[obj]["pre"], grad[obj]["post"])
        print(f"{obj}: gradient pre={np.nanmean(grad[obj]['pre']):+.2f} post={np.nanmean(grad[obj]['post']):+.2f} "
              f"Δ(post-pre)={gp:+.2f} p={pp:.3f} n={nn}")
    dga = grad["agg"]["post"] - grad["agg"]["pre"]
    dgi = grad["inert"]["post"] - grad["inert"]["pre"]
    m = np.isfinite(dga) & np.isfinite(dgi)
    dd = (dga - dgi)[m]
    if m.sum() >= 3:
        print(f"\nLearning specificity (agg gradient change - inert gradient change): mean={np.mean(dd):+.2f} "
              f"p={wilcoxon_signed_rank_p_value(dd)[0]:.3f} n={int(m.sum())}")
    _, p_pre, _ = paired(np.zeros_like(grad["agg"]["pre"]), grad["agg"]["pre"])
    print(f"Aggressive gradient PRE alone (innate check): mean={np.nanmean(grad['agg']['pre']):+.2f} vs 0, p={p_pre:.3f}")

    fig.text(0.5, -0.02, f"Aggressive near-far peak gradient: pre {np.nanmean(grad['agg']['pre']):+.2f}, "
             f"post {np.nanmean(grad['agg']['post']):+.2f} mm/s.", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_bout_vigor_prepost_{args.tag}.png"
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
