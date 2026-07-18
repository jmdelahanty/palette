#!/usr/bin/env python
"""Reproducible: per-fish GoodCopBadCop scorecard (learning index + avoidance metrics).

There is no stored "learner/non-learner" label anywhere (not in the zarr, not in the
registry). This assembles the per-recording individual-difference metrics from the
per-trial and per-bout tables, one row per fish, so individual variation is visible and
correlatable. Columns:

  LEARNING (the only thing training actually changes -- the flee->freeze habituation):
    esc_delta   escape rate late(trials>=5) - early(1-2)   (learner direction: < 0)
    frz_delta   freeze fraction late - early               (learner direction: > 0)
  ACUTE AVOIDANCE (reflexive, chase-driven):
    escape_fold chase escape rate / pre escape rate        (bigger = stronger flight)
  SPATIAL AVOIDANCE (near-shell steer-wider, object minus wall-matched virtual):
    steer_pre / steer_post   near-shell (8-22mm) avoidance steering excess
    steer_learned            steer_post - steer_pre        (innate vs learned check)
  LEARNED SPATIAL AVOIDANCE (near-object dwell vs wall-matched virtual -- the heatmap read):
    occ_avoid_pre / occ_avoid_post   P(near virtual) - P(near object), <15mm, per epoch
    occ_learned                      occ_avoid_post - occ_avoid_pre  (> 0 = avoids MORE post)
  ORIENTATION:
    lat_excess_chase  near (<25mm) chase lateral-keeping, object minus virtual

The cohort mean of occ_learned is ~null (spatial avoidance is innate on average), but
INDIVIDUAL fish vary -- some genuinely develop post-training avoidance of the object's
zone (visible in their occupancy heatmaps). Per-fish metrics exist to surface exactly that
heterogeneity, which the cohort mean hides.

CAVEATS (read before using): per-fish estimates come from ~8 chase trials and are noisy;
do NOT threshold these into a binary learner label -- ~86% of fish habituate, so there is
little genuine learner/non-learner variance. Use the continuous indices as covariates,
with a session random effect, not as individual verdicts.

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_per_fish        # table + CSV + figure

Writes a CSV + scatter to $PALETTE_RECORDINGS_ROOT/figures. Reads the canonical registry.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fisheye.analysis.goodcopbadcop_common import figures_dir, latest, nav, resolve_cohort
from fisheye.analysis.analyze_goodcopbadcop_escape import load as escape_load, escape_rate
from fisheye.analysis.analyze_goodcopbadcop_radial_turn_direction import load_rows as steer_load, _shell_mean
from fisheye.analysis.analyze_goodcopbadcop_lateral_gaze import load as gaze_load, sector_fractions

NEAR_SHELL = (8.0, 22.0)
MIN_SHELL_BOUTS = 5
NEAR_MM = 25.0
EARLY = (1, 2)
LATE = 5


def _early_late(dmap):
    e = [dmap[k] for k in EARLY if k in dmap and np.isfinite(dmap[k])]
    l = [dmap[k] for k in dmap if k >= LATE and np.isfinite(dmap[k])]
    return (np.mean(e), np.mean(l)) if e and l else (np.nan, np.nan)


def learning_index(zp):
    r = zarr.open_group(zp, mode="r")
    cd = latest(nav(r, ["analysis", "chaser_distance_runs"]))
    et = latest(cd["chaser_escape_events"])["trials"]
    o = np.asarray(et["ordinal"][:], int)
    esc = np.asarray(et["escape_rate_per_valid_s"][:], float)
    ef = latest(cd["chaser_escape_freeze"])
    fo = np.asarray(ef["trials"]["trial_ordinal"][:], int)
    ff = np.asarray(ef["trial_metrics"]["freeze_low_speed_fraction"][:], float)
    ee, el = _early_late({int(a): b for a, b in zip(o, esc)})
    fe, fl = _early_late({int(a): b for a, b in zip(fo, ff)})
    return {"esc_early": ee, "esc_late": el, "esc_delta": el - ee,
            "frz_early": fe, "frz_late": fl, "frz_delta": fl - fe}


def escape_fold(zp):
    d = escape_load(zp)
    pre = escape_rate(d, "pre", d["peak_smoothed"])
    chase = escape_rate(d, "chase", d["peak_smoothed"])
    fold = chase / pre if (np.isfinite(pre) and pre > 0) else np.nan
    return {"esc_rate_pre": pre, "esc_rate_chase": chase, "escape_fold": fold}


def steering_excess(zp):
    row = steer_load(zp)
    if row is None:
        return {"steer_pre": np.nan, "steer_post": np.nan, "steer_learned": np.nan}
    obj_cols, vir_cols = row["cols"]("delta_predicted_miss_mm")
    out = {}
    for name, eidx in (("pre", 0), ("post", 2)):
        om, _ = _shell_mean(row["valid"], row["epoch"], obj_cols, eidx, NEAR_SHELL, MIN_SHELL_BOUTS)
        vm, _ = _shell_mean(row["valid"], row["epoch"], vir_cols, eidx, NEAR_SHELL, MIN_SHELL_BOUTS)
        out[f"steer_{name}"] = (om - vm) if (np.isfinite(om) and np.isfinite(vm)) else np.nan
    out["steer_learned"] = (out["steer_post"] - out["steer_pre"]
                            if np.isfinite(out["steer_post"]) and np.isfinite(out["steer_pre"]) else np.nan)
    return out


def lateral_excess_chase(zp):
    d = gaze_load(zp)
    if "chase" not in d["epochs"]:
        return {"lat_excess_chase": np.nan}
    rng = d["epochs"]["chase"]
    _, on = sector_fractions(d["bearings"]["object"], d["valid"]["object"], d["dists"]["object"], rng, NEAR_MM)
    virt = [sector_fractions(d["bearings"][k], d["valid"][k], d["dists"][k], rng, NEAR_MM)[1][0]
            for k in d["bearings"] if k.startswith("virt_")]
    lat = on[0] - np.nanmean(virt) if np.isfinite(on[0]) else np.nan
    return {"lat_excess_chase": lat}


def spatial_avoidance(zp):
    """Per-fish learned SPATIAL avoidance: near-object dwell vs a wall-matched virtual.

    occ_avoid = P(near virtual, <15mm) - P(near object, <15mm) in an epoch: positive means
    the fish spends LESS time near the object's actual location than near a rotated no-object
    reference at the same wall proximity -- object-specific spatial avoidance. occ_learned =
    post - pre: positive = learned to avoid the object's location (the "post heatmap stays
    away" pattern). The cohort mean of this is ~null, but individual fish vary -- that
    variance is the point of a per-fish metric.
    """
    thr = 15.0
    d = gaze_load(zp)
    out = {}
    for name in ("pre", "post"):
        if name not in d["epochs"]:
            out[f"occ_avoid_{name}"] = np.nan
            continue
        s, e = d["epochs"][name]
        ov, od = d["valid"]["object"], d["dists"]["object"]
        mo = ov.copy(); mo[:s] = False; mo[e + 1:] = False; mo &= np.isfinite(od)
        obj_near = float(np.mean(od[mo] < thr)) if mo.sum() > 100 else np.nan
        vns = []
        for k in d["dists"]:
            if not k.startswith("virt_"):
                continue
            vv, vd = d["valid"][k], d["dists"][k]
            mv = vv.copy(); mv[:s] = False; mv[e + 1:] = False; mv &= np.isfinite(vd)
            if mv.sum() > 100:
                vns.append(float(np.mean(vd[mv] < thr)))
        vir_near = np.nanmean(vns) if vns else np.nan
        out[f"occ_avoid_{name}"] = (vir_near - obj_near) if (np.isfinite(obj_near) and np.isfinite(vir_near)) else np.nan
    out["occ_learned"] = (out["occ_avoid_post"] - out["occ_avoid_pre"]
                          if np.isfinite(out.get("occ_avoid_post", np.nan)) and np.isfinite(out.get("occ_avoid_pre", np.nan)) else np.nan)
    return out


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 6:
        return np.nan, int(m.sum())
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-18")
    args = ap.parse_args()
    out_dir = args.out_dir or figures_dir()

    metric_fns = (learning_index, escape_fold, steering_excess, lateral_excess_chase, spatial_avoidance)
    cols = ["esc_early", "esc_late", "esc_delta", "frz_early", "frz_late", "frz_delta",
            "esc_rate_pre", "esc_rate_chase", "escape_fold",
            "steer_pre", "steer_post", "steer_learned", "lat_excess_chase",
            "occ_avoid_pre", "occ_avoid_post", "occ_learned"]
    rows = []
    for rid, zp in resolve_cohort():
        rec = {"recording_id": rid, "session": re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)", rid).group(1)}
        got_any = False
        for fn in metric_fns:
            try:
                rec.update(fn(zp))
                got_any = True
            except Exception:
                pass
        if got_any:
            for c in cols:
                rec.setdefault(c, np.nan)
            rows.append(rec)
    n = len(rows)

    # CSV
    csv_path = out_dir / f"goodcopbadcop_per_fish_metrics_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "session"] + cols)
        for r in rows:
            w.writerow([r["recording_id"], r["session"]] + [f"{r[c]:.4f}" if np.isfinite(r[c]) else "" for c in cols])
    print(f"n={n} fish. wrote {csv_path}\n")

    def col(name):
        return np.array([r[name] for r in rows], float)

    print(f"{'recording':34s} {'esc_delta':>9s} {'frz_delta':>9s} {'esc_fold':>8s} {'steer_post':>10s} {'occ_learned':>11s}")
    for r in sorted(rows, key=lambda x: (np.nan_to_num(x["occ_learned"], nan=-9))):
        def s(c, w=9, p=3):
            return (f"{r[c]:>{w}.{p}f}" if np.isfinite(r[c]) else " " * (w - 1) + ".")
        print(f"{r['recording_id'][:34]:34s} {s('esc_delta')} {s('frz_delta')} {s('escape_fold',8,2)} {s('steer_post',10)} {s('occ_learned',11)}")

    print("\n--- learner-direction counts (per fish) ---")
    ed, fd, ol = col("esc_delta"), col("frz_delta"), col("occ_learned")
    print(f"  escape decreased (learner):          {int((ed < 0).sum())}/{np.isfinite(ed).sum()}")
    print(f"  freeze increased (learner):          {int((fd > 0).sum())}/{np.isfinite(fd).sum()}")
    print(f"  learned SPATIAL avoidance (occ_learned>0): {int((ol > 0).sum())}/{np.isfinite(ol).sum()}  "
          f"(cohort mean Δ={np.nanmean(ol):+.3f}) -- individual variation is the point")

    print("\n--- do learners avoid more? (Spearman across fish; noisy, use a session RE for real inference) ---")
    for lname, lcol in (("freeze learning (frz_delta)", fd), ("escape learning (-esc_delta)", -ed)):
        for aname, acol in (("steer_post", col("steer_post")), ("occ_learned", ol),
                            ("lat_excess_chase", col("lat_excess_chase"))):
            rho, nn = spearman(lcol, acol)
            print(f"  {lname:28s} vs {aname:16s}: rho={rho:+.2f} (n={nn})")

    # figure: learning vs retained avoidance steering, colored by session
    sessions = sorted({r["session"] for r in rows})
    cmap = plt.cm.tab10(np.linspace(0, 1, len(sessions)))
    sc_color = {s: cmap[i] for i, s in enumerate(sessions)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # Panel A: freeze learning vs learned spatial avoidance
    ax = axes[0]
    for r in rows:
        if np.isfinite(r["frz_delta"]) and np.isfinite(r["occ_learned"]):
            ax.scatter(r["frz_delta"], r["occ_learned"], color=sc_color[r["session"]], s=36, alpha=0.85)
    ax.axvline(0, color="#ccc", lw=1); ax.axhline(0, color="#ccc", lw=1)
    rho, nn = spearman(fd, ol)
    ax.set_xlabel("freeze learning index (late − early)")
    ax.set_ylabel("learned spatial avoidance (occ post − pre)")
    ax.set_title(f"Learned freeze vs learned spatial avoidance\n(Spearman rho={rho:+.2f}, n={nn})", fontsize=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    # Panel B: pre vs post spatial avoidance — above the diagonal = developed avoidance
    ax = axes[1]
    ap, apo = col("occ_avoid_pre"), col("occ_avoid_post")
    lim = np.nanmax(np.abs(np.concatenate([ap, apo]))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="#bbb", lw=1)
    for r in rows:
        if np.isfinite(r["occ_avoid_pre"]) and np.isfinite(r["occ_avoid_post"]):
            ax.scatter(r["occ_avoid_pre"], r["occ_avoid_post"], color=sc_color[r["session"]], s=36, alpha=0.85)
    ax.set_xlabel("spatial avoidance PRE (virtual − object near-dwell)")
    ax.set_ylabel("spatial avoidance POST")
    ax.set_title("Above diagonal = fish that avoid MORE post\n(individual learned spatial avoidance)", fontsize=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=sc_color[s], label=s[:10]) for s in sessions]
    ax.legend(handles=handles, frameon=False, fontsize=7, title="session", loc="best")
    fig.suptitle(f"Per-fish learning vs avoidance (n={n}) — cohort mean is null but individuals vary (noisy, ~8 trials)",
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_per_fish_{args.tag}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
