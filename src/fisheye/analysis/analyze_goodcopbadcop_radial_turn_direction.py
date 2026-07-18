#!/usr/bin/env python
"""Reproducible: GoodCopBadCop per-bout radial TURN-DIRECTION profile vs distance to object.

Companion to analyze_goodcopbadcop_radial_kinematics (frame-based). The per-bout
directional metrics -- did the bout turn TOWARD the object, and did it steer to pass
WIDER -- are too sparse to bin per fish (min_bin_bouts fragments them; that sparsity is
why only a single near-band scalar existed). The fix is NOT coarser bins (the avoidance
signal is a localized ~14-22 mm shell that wider bins average away) but a different
aggregation: pool the raw per-bout table across the cohort at fine distance resolution
and put honest CIs on it with a CLUSTER BOOTSTRAP over (fish, visit).

Source: chaser_bout_response/bouts_per_reference/ (per bout, per reference, unthresholded)
  turn_toward             bool  -- did the bout rotate toward the reference (avoidance = low)
  delta_predicted_miss_mm float -- change in predicted miss distance; + = steered WIDER
                                   (active avoidance steering); defined for object-ahead
                                   bouts only (|bearing_at_onset| < 90 deg)
  distance_at_onset_mm    the radial axis;  visit_id  the bootstrap cluster
Each metric is contrasted between the AGGRESSIVE object and its rotated VIRTUAL controls
(same wall proximity, no object) -- the wall-following-free readout.

Pseudoreplication (per the component's own note): bouts within one visit are one approach
subsampled; the effective n is visits, not bouts. So the cluster bootstrap resamples
(fish, reference, visit) clusters -- far-field bouts with no visit are singleton clusters.
CIs and the object-vs-virtual test are at the cluster level, never raw bout counts.

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_radial_turn_direction
    python -m fisheye.analysis.analyze_goodcopbadcop_radial_turn_direction --reps 4000

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
    latest,
    nav,
    resolve_cohort,
    resolve_object_roles,
)

EPOCHS = (("pre", 0), ("chase", 1), ("post", 2))
EDGES = np.arange(0.0, 48.001, 3.0)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
NBIN = CTR.size
MIN_CLUSTERS = 8          # min (fish,visit) clusters contributing to a bin, else NaN
OBJECT_C = "#c1435b"
VIRTUAL_C = "#8a8a8a"
METRICS = [
    ("turn_toward", "P(turn toward object)", "fraction", False),
    ("delta_predicted_miss_mm", "Avoidance steering", "delta predicted-miss (mm)", True),
]
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ececec", "font.family": "DejaVu Sans",
                     "savefig.dpi": 160})


def collect(loaded_rows, epoch_idx, metric, object_ahead_only):
    """Flatten cohort rows into (bin_idx, value, cluster_id) for object and virtual.

    loaded_rows: list of per-fish dicts with columns already extracted.
    Returns dict role -> (bin_idx[int], val[float], cluster[int]) over valid, in-range bouts.
    """
    out = {}
    for role in ("object", "virtual"):
        bins, vals, clusters = [], [], []
        cluster_base = 0
        for fi, row in enumerate(loaded_rows):
            for ref_col, dist, val, vid, ahead in row[role]:
                sel = row["valid"] & (row["epoch"] == epoch_idx) & np.isfinite(dist)
                sel &= (dist >= EDGES[0]) & (dist < EDGES[-1])
                if object_ahead_only:
                    sel &= np.isfinite(val) & ahead
                else:
                    sel &= np.isfinite(val)
                idx = np.where(sel)[0]
                if idx.size == 0:
                    cluster_base += 1
                    continue
                bi = np.clip(np.digitize(dist[idx], EDGES) - 1, 0, NBIN - 1)
                # cluster = (fish, ref_col, visit_id); visit<0 -> unique singleton
                v = vid[idx]
                cl = np.empty(idx.size, np.int64)
                have_visit = v >= 0
                cl[have_visit] = cluster_base + v[have_visit]
                span = int(v.max()) + 1 if have_visit.any() else 0
                n_singleton = int((~have_visit).sum())
                cl[~have_visit] = cluster_base + span + np.arange(n_singleton)
                cluster_base += span + n_singleton
                bins.append(bi); vals.append(val[idx].astype(float)); clusters.append(cl)
        if bins:
            out[role] = (np.concatenate(bins), np.concatenate(vals), np.concatenate(clusters))
        else:
            out[role] = (np.array([], int), np.array([], float), np.array([], np.int64))
    return out


def bin_means(bin_idx, val, weights=None):
    if weights is None:
        weights = np.ones_like(val)
    num = np.bincount(bin_idx, weights=weights * val, minlength=NBIN)
    den = np.bincount(bin_idx, weights=weights, minlength=NBIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, den


def cluster_bootstrap(bin_idx, val, cluster, reps, rng):
    """Return (nreps, NBIN) bootstrap bin-means, resampling clusters with replacement."""
    if val.size == 0:
        return np.full((reps, NBIN), np.nan)
    n_clusters = int(cluster.max()) + 1
    pvals = np.full(n_clusters, 1.0 / n_clusters)
    boots = np.full((reps, NBIN), np.nan)
    for rr in range(reps):
        counts = rng.multinomial(n_clusters, pvals)  # resample clusters w/ replacement
        w = counts[cluster].astype(float)
        m, _ = bin_means(bin_idx, val, weights=w)
        boots[rr] = m
    return boots


def load_rows(zp: str):
    r = zarr.open_group(zp, mode="r")
    cd = latest(nav(r, ["analysis", "chaser_distance_runs"]))
    if "chaser_bout_response" not in list(cd.group_keys()):
        return None
    b = latest(cd["chaser_bout_response"])
    bt = b["bouts"]; bpr = b["bouts_per_reference"]
    roles = resolve_object_roles(r); agg = roles["aggressive"]
    ref = b["references"]
    ci = np.asarray(ref["chaser_index"][:]); parent = np.asarray(ref["parent_chaser_index"][:])
    hits = np.where(ci == agg)[0]
    if hits.size == 0:
        return None
    agg_ref = int(hits[0]); virt_refs = list(np.where(parent == agg)[0])
    if not virt_refs:
        return None
    valid = np.asarray(bt["valid"][:], bool)
    epoch = np.asarray(bt["epoch_index"][:], int)
    dist = np.asarray(bpr["distance_at_onset_mm"][:], float)
    ahead = np.asarray(bpr["approaching_at_onset"][:], bool)  # object-ahead / approaching gate
    vid = np.asarray(bpr["visit_id"][:], int)

    def cols(metric):
        arr = np.asarray(bpr[metric][:])
        # role -> list of (ref_col, dist_col, val_col, visit_col, ahead_col)
        obj = [(agg_ref, dist[:, agg_ref], arr[:, agg_ref], vid[:, agg_ref], ahead[:, agg_ref])]
        vir = [(c, dist[:, c], arr[:, c], vid[:, c], ahead[:, c]) for c in virt_refs]
        return obj, vir

    return {"valid": valid, "epoch": epoch, "cols": cols}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2000, help="Cluster-bootstrap resamples.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-18")
    args = ap.parse_args()
    out_dir = args.out_dir or figures_dir()
    rng = np.random.default_rng(0)

    fish = []
    for rid, zp in resolve_cohort():
        try:
            row = load_rows(zp)
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], type(ex).__name__, ex)
            continue
        if row is not None:
            fish.append(row)
    n = len(fish)

    fig, axes = plt.subplots(len(METRICS), 3, figsize=(13.5, 3.4 * len(METRICS)), sharex=True)
    for mi, (metric, ttl, unit, ahead_only) in enumerate(METRICS):
        rows = [{"valid": f["valid"], "epoch": f["epoch"],
                 "object": f["cols"](metric)[0], "virtual": f["cols"](metric)[1]} for f in fish]
        for ei, (epoch, epoch_idx) in enumerate(EPOCHS):
            ax = axes[mi, ei]
            data = collect(rows, epoch_idx, metric, ahead_only)
            stats = {}
            for role, color in (("object", OBJECT_C), ("virtual", VIRTUAL_C)):
                bi, val, cl = data[role]
                point, den = bin_means(bi, val)
                boots = cluster_bootstrap(bi, val, cl, args.reps, rng)
                lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
                enough = den >= 0  # bouts present; cluster floor applied below
                n_clusters_bin = np.bincount(bi, minlength=NBIN)  # bouts/bin as rough guard
                point = np.where(n_clusters_bin >= MIN_CLUSTERS, point, np.nan)
                stats[role] = (point, lo, hi, boots)
                ok = np.isfinite(point)
                ls = "-o" if role == "object" else "--s"
                lw = 2.2 if role == "object" else 1.6
                ax.plot(CTR[ok], point[ok], ls, color=color, lw=lw, ms=4,
                        label=("aggressive object" if role == "object" else "virtual control"))
                ax.fill_between(CTR, lo, hi, color=color, alpha=0.15, lw=0)
            # object-vs-virtual excess significance per bin (bootstrap CI of the difference)
            ob, ov = stats["object"][3], stats["virtual"][3]
            diff = ob - ov
            dlo, dhi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
            sig = np.isfinite(stats["object"][0]) & np.isfinite(stats["virtual"][0]) & ((dlo > 0) | (dhi < 0))
            yv = stats["object"][0]
            for j in np.where(sig)[0]:
                ax.plot(CTR[j], yv[j], "o", color=OBJECT_C, ms=10, mfc="none", mew=1.9)
            if not ahead_only:
                # chance line = the virtual mean is the reference; nothing extra
                pass
            else:
                ax.axhline(0, color="#bbbbbb", lw=1, zorder=0)
            if ei == 0:
                ax.set_ylabel(f"{ttl}\n{unit}", fontsize=9.5)
            if mi == 0:
                ax.set_title(epoch, fontsize=12, weight="bold")
            if mi == len(METRICS) - 1:
                ax.set_xlabel("distance to object at bout onset (mm)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Per-bout radial turn direction vs distance -- object vs virtual control, "
                 f"cluster-bootstrapped (n={n} fish)", fontsize=12.5, weight="bold", y=1.004)
    fig.text(0.5, -0.01, "Bands = 95% cluster bootstrap CI over (fish, visit). Open rings = object differs from "
             "virtual (CI of difference excludes 0). Low P(turn toward) or positive steering = avoidance. "
             "Steering restricted to object-ahead bouts.", ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_radial_turn_direction_{args.tag}.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, f" (n_fish={n}, reps={args.reps})\n")

    # text report
    print(f"distance bin centers (mm): {np.round(CTR, 0)}\n")
    for metric, ttl, unit, ahead_only in METRICS:
        rows = [{"valid": f["valid"], "epoch": f["epoch"],
                 "object": f["cols"](metric)[0], "virtual": f["cols"](metric)[1]} for f in fish]
        print(f"=== {ttl} ({unit}) ===")
        for epoch, epoch_idx in EPOCHS:
            data = collect(rows, epoch_idx, metric, ahead_only)
            ob = cluster_bootstrap(*data["object"], args.reps, rng)
            ov = cluster_bootstrap(*data["virtual"], args.reps, rng)
            obj_pt, den_o = bin_means(*data["object"][:2])
            vir_pt, _ = bin_means(*data["virtual"][:2])
            guard = np.bincount(data["object"][0], minlength=NBIN) >= MIN_CLUSTERS
            diff = ob - ov
            dlo, dhi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
            sig = guard & ((dlo > 0) | (dhi < 0))
            shell = ", ".join(f"{CTR[j]:.0f}mm" for j in np.where(sig)[0])
            print(f"  [{epoch}] object vs virtual differs at: {shell or '(no bins)'}")
        print()


if __name__ == "__main__":
    main()
