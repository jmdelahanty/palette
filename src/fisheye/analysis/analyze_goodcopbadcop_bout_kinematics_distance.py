#!/usr/bin/env python
"""Reproducible: GoodCopBadCop bout kinematics as a function of onset distance to object.

Static epochs (pre + post). For each bout, distance-to-object at onset is binned; the
median peak speed / |turn| / duration / path length per bin is plotted, aggressive vs
inert. Tests whether the fish bouts more vigorously when NEAR the aggressive object than
near the inert one. Bout-level -> robust, above the raw-speed noise floor.

Run (palette env):
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_bout_kinematics_distance --exploratory-only

Key result (full canonical cohort, n=32): peak-speed near-minus-far, aggressive vs inert
(diff-in-diff) ~+1.24 mm/s, p~0.063 -- MARGINAL (the n=12 June-14 slice gave +2.7,
p~0.034; the effect shrank on the full cohort). See analyze_goodcopbadcop_bout_vigor_prepost
for the pre/post split; the near-object vigor gradient is weak/suggestive at full n.
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
    parse_standalone_exploratory_args,
    resolve_cohort,
    save_standalone_exploratory_figure,
)
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

EDGES = np.array([3, 6, 9, 13, 18, 25, 35, 50], float)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
AGG_C = "#c1435b"
INERT_C = "#3a7ca5"
NEAR = (7, 18)
FAR = (25, 50)
MIN_BOUTS = 15
KIN_FIELDS = ("peak_speed_mm_s", "turn_deg", "duration_s", "path_length_mm")
PANELS = [("peak_speed_mm_s", "Peak speed", "mm/s"), ("turn_deg", "|Turn angle|", "deg"),
          ("duration_s", "Duration", "s"), ("path_length_mm", "Path length", "mm")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "font.family": "DejaVu Sans",
                     "savefig.dpi": 160})


def load(zp: str):
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_bout_response")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-17")
    args = parse_standalone_exploratory_args(
        ap,
        analysis_id="goodcopbadcop_bout_kinematics_distance",
    )
    out_dir = args.out_dir or figures_dir()

    pool = {obj: {k: {"d": [], "v": []} for k in KIN_FIELDS} for obj in ("agg", "inert")}
    near_far = {obj: {"near": [], "far": []} for obj in ("agg", "inert")}
    n = 0
    for rid, zp in resolve_cohort():
        try:
            dist, bs, kin, roles = load(zp)
        except ChaserDistanceReadError:
            raise
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        n += 1
        for obj, idx in (("agg", roles["aggressive"]), ("inert", roles["inert"])):
            dob = dist[bs, idx]
            for k in kin:
                ok = np.isfinite(dob) & np.isfinite(kin[k])
                pool[obj][k]["d"].append(dob[ok])
                pool[obj][k]["v"].append(kin[k][ok])
            pk = kin["peak_speed_mm_s"]
            nm = np.isfinite(dob) & (dob >= NEAR[0]) & (dob <= NEAR[1])
            fm = np.isfinite(dob) & (dob >= FAR[0]) & (dob <= FAR[1])
            near_far[obj]["near"].append(np.median(pk[nm]) if nm.sum() >= MIN_BOUTS else np.nan)
            near_far[obj]["far"].append(np.median(pk[fm]) if fm.sum() >= MIN_BOUTS else np.nan)

    def curve(obj, k):
        d = np.concatenate(pool[obj][k]["d"])
        v = np.concatenate(pool[obj][k]["v"])
        med = np.full(CTR.size, np.nan)
        cnt = np.zeros(CTR.size, int)
        for i in range(CTR.size):
            m = (d >= EDGES[i]) & (d < EDGES[i + 1])
            cnt[i] = m.sum()
            if m.sum() >= MIN_BOUTS:
                med[i] = np.median(v[m])
        return med, cnt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, (k, ttl, unit) in zip(axes, PANELS):
        for obj, col, lab in (("agg", AGG_C, "aggressive"), ("inert", INERT_C, "inert")):
            med, _ = curve(obj, k)
            ok = np.isfinite(med)
            ax.plot(CTR[ok], med[ok], "-o", color=col, lw=2.2, ms=5, label=lab)
        ax.axvspan(*NEAR, color="#f2c94c", alpha=0.12, zorder=0)
        ax.set_title(ttl, fontsize=11.5, weight="bold")
        ax.set_xlabel("onset distance to object (mm)")
        ax.set_ylabel(unit)
        ax.set_xlim(3, 52)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    an = np.array(near_far["agg"]["near"], float) - np.array(near_far["agg"]["far"], float)
    inn = np.array(near_far["inert"]["near"], float) - np.array(near_far["inert"]["far"], float)
    m = np.isfinite(an) & np.isfinite(inn)
    dd_val = np.mean((an - inn)[m])
    dd_p = wilcoxon_signed_rank_p_value((an - inn)[m])[0]

    fig.suptitle(f"Bout kinematics vs onset distance to object -- static epochs, pooled bouts (n={n})",
                 fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02, f"Amber = 7-18 mm shell. Peak-speed near-minus-far, aggressive vs inert "
             f"(diff-in-diff): Δ={dd_val:+.1f} mm/s, p={dd_p:.3f}, n={int(m.sum())}. Bout-level (robust).",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_bout_kinematics_distance_{args.tag}.png"
    out, _ = save_standalone_exploratory_figure(
        fig,
        out,
        analysis_id="goodcopbadcop_bout_kinematics_distance",
        bbox_inches="tight",
    )
    print("wrote", out)
    print(f"n={n}  diff-in-diff peak (near-far, agg-inert) Δ={dd_val:+.2f} mm/s p={dd_p:.3f} n={int(m.sum())}")
    for obj in ("agg", "inert"):
        med, cnt = curve(obj, "peak_speed_mm_s")
        tmed, _ = curve(obj, "turn_deg")
        print(f"  {obj}: peak by dist {np.round(med, 1)}  |turn| {np.round(tmed, 0)}  (n/bin {cnt})")


if __name__ == "__main__":
    main()
