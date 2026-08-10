#!/usr/bin/env python
"""Reproducible: GoodCopBadCop radial bout-kinematics profile vs distance to the object.

Goes beyond the single near-band ("reactive ring") scalar: profiles how the fish moves
as a continuous function of distance to the AGGRESSIVE object, across the full 0-60 mm
range, per epoch, and -- critically -- against the object's own VIRTUAL controls (its
position rotated about the arena centre, so identical wall proximity but no object). The
object-minus-virtual contrast is the wall-following-free avoidance readout.

All inputs are pre-computed per recording in `chaser_bout_response`:
  binned/ (epoch x reference x distance_bin, 9 bins over [0,4,8,12,16,20,25,30,40,60] mm)
    radial_velocity_mm_s   (+ = closing on object, - = moving away)   <- headline
    bout_rate_per_min      (bouting density vs distance)
    tangential_speed_mm_s  (lateral / circling motion vs distance)
  object_vs_virtual/steering_excess_by_band (epoch x object x distance_bin)
    active avoidance steering (delta predicted-miss), already object-minus-virtual; the
    component author's PRIMARY read (localized, decays with distance if real).

Aggregation is FISH-LEVEL: each fish contributes one object curve and one mean-virtual
curve per epoch; the cohort mean +/- SEM and the per-bin object-vs-virtual test are across
fish (fish = the unit). This sidesteps the bout pseudoreplication the component warns about
(bouts within a visit are one approach subsampled; near_visit_count is the effective n).
Frame-based metrics are used because the per-bout summaries (turn, peak) are too sparse
per distance bin per reference to populate -- that sparsity is exactly why only a single
near-band scalar existed before.

Run (palette env):
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_radial_kinematics --exploratory-only

Reads the canonical registry (see goodcopbadcop_common); n=32 usable (June 14 + June 21).
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

EPOCH_LABELS = ("pre", "chase", "post")  # binned axis order: pre_event, training_event, post_event
OBJECT_C = "#c1435b"
VIRTUAL_C = "#8a8a8a"
EXCESS_C = "#1b3a6b"
# Frame-based binned metrics (object vs virtual), + the precomputed excess-by-band.
FRAME_METRICS = [
    ("radial_velocity_mm_s", "Radial velocity", "mm/s  (+toward / -away)"),
    ("bout_rate_per_min", "Bout rate", "bouts/min"),
    ("tangential_speed_mm_s", "Tangential speed", "mm/s"),
]
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ececec", "font.family": "DejaVu Sans",
                     "savefig.dpi": 160})


def load(zp: str):
    """Per-fish object and mean-virtual radial curves for the aggressive object.

    Returns dict metric -> {'object': (3,9), 'virtual': (3,9)} plus
    'steering_excess': (3,9) and the distance centres, or None if unavailable.
    """
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_bout_response")


def cohort_stack(loaded):
    """Stack per-fish curves -> arrays keyed for aggregation."""
    centers = loaded[0]["centers"]
    stacks = {key: {"object": [], "virtual": []} for key, _, _ in FRAME_METRICS}
    steer = []
    for d in loaded:
        for key, _, _ in FRAME_METRICS:
            stacks[key]["object"].append(d["metrics"][key]["object"])
            stacks[key]["virtual"].append(d["metrics"][key]["virtual"])
        steer.append(d["steering_excess"])
    for key in stacks:
        stacks[key]["object"] = np.array(stacks[key]["object"])   # (nfish, 3, 9)
        stacks[key]["virtual"] = np.array(stacks[key]["virtual"])
    return centers, stacks, np.array(steer)  # steer: (nfish, 3, 9)


def mean_sem(a, axis=0):
    n = np.sum(np.isfinite(a), axis=axis)
    m = np.nanmean(a, axis=axis)
    s = np.nanstd(a, axis=axis) / np.sqrt(np.maximum(n, 1))
    return m, s, n


def per_bin_excess_p(obj, virt):
    """Wilcoxon of (object - virtual) vs 0 across fish, per distance bin."""
    diff = obj - virt  # (nfish, 9)
    ps = np.full(diff.shape[1], np.nan)
    for j in range(diff.shape[1]):
        col = diff[:, j][np.isfinite(diff[:, j])]
        if col.size >= 6:
            ps[j] = wilcoxon_signed_rank_p_value(col)[0]
    return ps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-18")
    args = parse_standalone_exploratory_args(
        ap,
        analysis_id="goodcopbadcop_radial_kinematics",
    )
    out_dir = args.out_dir or figures_dir()

    loaded = []
    for rid, zp in resolve_cohort():
        try:
            d = load(zp)
        except ChaserDistanceReadError:
            raise
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], type(ex).__name__, ex)
            continue
        if d is not None:
            loaded.append(d)
    n = len(loaded)
    centers, stacks, steer = cohort_stack(loaded)
    x = centers

    rows = FRAME_METRICS + [("steering_excess", "Avoidance steering (excess vs virtual)", "mm / bout")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.1 * len(rows)), sharex=True)
    for ei, epoch in enumerate(EPOCH_LABELS):
        for ri, (key, ttl, unit) in enumerate(rows):
            ax = axes[ri, ei]
            if key == "steering_excess":
                m, s, _ = mean_sem(steer[:, ei, :])
                ax.axhline(0, color="#bbbbbb", lw=1, zorder=0)
                ax.errorbar(x, m, yerr=s, fmt="-o", color=EXCESS_C, lw=2.2, ms=4, capsize=2)
                ps = np.full(x.size, np.nan)
                for j in range(x.size):
                    col = steer[:, ei, j][np.isfinite(steer[:, ei, j])]
                    if col.size >= 6:
                        ps[j] = wilcoxon_signed_rank_p_value(col)[0]
                for j in np.where(ps < 0.05)[0]:
                    ax.plot(x[j], m[j], "o", color=EXCESS_C, ms=9, mfc="none", mew=1.8)
            else:
                obj = stacks[key]["object"][:, ei, :]
                virt = stacks[key]["virtual"][:, ei, :]
                mo, so, _ = mean_sem(obj)
                mv, sv, _ = mean_sem(virt)
                ax.errorbar(x, mo, yerr=so, fmt="-o", color=OBJECT_C, lw=2.2, ms=4, capsize=2, label="aggressive object")
                ax.errorbar(x, mv, yerr=sv, fmt="--s", color=VIRTUAL_C, lw=1.6, ms=3.5, capsize=2, label="virtual control")
                if key == "radial_velocity_mm_s":
                    ax.axhline(0, color="#bbbbbb", lw=1, zorder=0)
                ps = per_bin_excess_p(obj, virt)
                for j in np.where(ps < 0.05)[0]:
                    ax.plot(x[j], mo[j], "o", color=OBJECT_C, ms=9, mfc="none", mew=1.8)
            if ei == 0:
                ax.set_ylabel(f"{ttl}\n{unit}", fontsize=9.5)
            if ri == 0:
                ax.set_title(epoch, fontsize=12, weight="bold")
            if ri == len(rows) - 1:
                ax.set_xlabel("distance to object (mm)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Radial bout kinematics vs distance to the aggressive object -- object vs virtual control "
                 f"(fish-level, n={n})", fontsize=13, weight="bold", y=1.005)
    fig.text(0.5, -0.01, "Open rings = distance bins where object differs from virtual control across fish "
             "(Wilcoxon p<0.05, fish = unit). Radial velocity <0 = moving away. Virtual = object rotated about "
             "arena centre (same wall proximity, no object).", ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_radial_kinematics_{args.tag}.png"
    out, _ = save_standalone_exploratory_figure(
        fig,
        out,
        analysis_id="goodcopbadcop_radial_kinematics",
        bbox_inches="tight",
    )
    print("wrote", out, f" (n_fish={n})\n")

    # text report: chase + post, headline radial velocity + steering
    print(f"distance bin centers (mm): {np.round(x, 0)}\n")
    for ei, epoch in enumerate(EPOCH_LABELS):
        if epoch == "pre":
            continue
        obj = stacks["radial_velocity_mm_s"]["object"][:, ei, :]
        virt = stacks["radial_velocity_mm_s"]["virtual"][:, ei, :]
        mo, _, _ = mean_sem(obj)
        mv, _, _ = mean_sem(virt)
        ps = per_bin_excess_p(obj, virt)
        print(f"[{epoch}] radial velocity (mm/s): + toward object, - away")
        print(f"    object : {np.round(mo, 1)}")
        print(f"    virtual: {np.round(mv, 1)}")
        print(f"    excess p (obj-virt, fish-level): {np.array([f'{p:.3f}' if np.isfinite(p) else '  . ' for p in ps])}")
        ms, _, _ = mean_sem(steer[:, ei, :])
        print(f"    steering excess (mm/bout): {np.round(ms, 2)}\n")


if __name__ == "__main__":
    main()
