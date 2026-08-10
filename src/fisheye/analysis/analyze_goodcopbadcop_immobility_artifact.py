#!/usr/bin/env python
"""Reproducible: the GoodCopBadCop mid-band immobility "avoidance" was a noise artifact.

This is the cautionary-tale analysis. The original readout -- P(fish speed < 1 mm/s)
in the 7-18 mm shell around the aggressive object, pre vs post -- looked like learned
avoidance (Δ+0.177, p=0.002). It was measuring the raw-centroid tracking noise floor
(~1.6 mm/s median), not behaviour. Two demonstrations, both on the same band/valid logic:

  recompute : the identical metric on RAW vs SMOOTHED speed. On smoothed the effect
              collapses (Δ~+0.004, p~0.85); inert is null; per-bout vigor is unchanged.
  sweep     : sweep the immobility threshold. A real "more still" effect stays positive
              at every threshold; the raw effect FLIPS SIGN once the threshold clears the
              ~1.6 mm/s noise floor (the tell that it was distributional noise, not stillness).

Run (palette env):
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_immobility_artifact --exploratory-only
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_immobility_artifact --exploratory-only --sweep-only

See docs/diagnostics/goodcopbadcop_behavior_synthesis_handoff_2026-07-17.md and the
memory note project_immobility_speed_artifact. The component fix landed as sun 7e931481
(chaser_response_regimes now classifies immobility on speed_smoothed_mm).
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.analysis.goodcopbadcop_common import (
    parse_standalone_exploratory_args,
    resolve_cohort,
)
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

MID_BAND_MM = (7.0, 18.0)
IMMOBILE_THR_MM_S = 1.0
SWEEP_THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
MIN_FRAMES = 150
MIN_BOUTS = 5


def load(zp: str) -> dict:
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_bout_response")


def band_immobile_frac(d, epoch, obj, key, thr) -> float:
    s, e = d["epochs"][epoch]
    sl = slice(s, e + 1)
    spd = d[key][sl]
    dc = d["dist"][sl, obj]
    ok = (d["moving_valid"][sl] & np.isfinite(spd) & d["chaser_valid"][sl, obj]
          & (dc >= MID_BAND_MM[0]) & (dc <= MID_BAND_MM[1]))
    return float((spd[ok] < thr).mean()) if ok.sum() >= MIN_FRAMES else np.nan


def band_epoch_durations(d, epoch, obj) -> np.ndarray:
    s, e = d["epochs"][epoch]
    sl = slice(s, e + 1)
    spd = d["speed"][sl]
    mv = d["moving_valid"][sl]
    dc = d["dist"][sl, obj]
    immobile = mv & np.isfinite(spd) & (spd < IMMOBILE_THR_MM_S)
    durs = []
    i, n = 0, immobile.size
    while i < n:
        if immobile[i]:
            j = i
            while j < n and immobile[j]:
                j += 1
            md = np.nanmedian(dc[i:j])
            if np.isfinite(md) and MID_BAND_MM[0] <= md <= MID_BAND_MM[1]:
                durs.append((j - i) / d["fps"])
            i = j
        else:
            i += 1
    return np.array(durs, float)


def bout_vigor(d, epoch, obj, field) -> float:
    s, e = d["epochs"][epoch]
    msk = (d["onsets"] >= s) & (d["onsets"] <= e)
    di = d["dist"][d["onsets"][msk], obj]
    in_band = (di >= MID_BAND_MM[0]) & (di <= MID_BAND_MM[1]) & np.isfinite(di)
    vals = d["bout"][field][msk][in_band]
    return float(np.median(vals)) if vals.size >= MIN_BOUTS else np.nan


def paired(pre, post):
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    m = np.isfinite(pre) & np.isfinite(post)
    dd = post[m] - pre[m]
    p = wilcoxon_signed_rank_p_value(dd)[0] if dd.size >= 3 else np.nan
    return dd, p, int(m.sum())


def run_recompute(loaded) -> None:
    keys = ("raw_pre", "raw_post", "af_pre", "af_post", "if_pre", "if_post",
            "dur_pre", "dur_post", "pk_pre", "pk_post", "path_pre", "path_post",
            "bdur_pre", "bdur_post")
    C = {k: [] for k in keys}
    for d in loaded:
        agg, inert = d["roles"]["aggressive"], d["roles"]["inert"]
        if "pre" not in d["epochs"] or "post" not in d["epochs"]:
            continue
        C["raw_pre"].append(band_immobile_frac(d, "pre", agg, "speed_raw", IMMOBILE_THR_MM_S))
        C["raw_post"].append(band_immobile_frac(d, "post", agg, "speed_raw", IMMOBILE_THR_MM_S))
        C["af_pre"].append(band_immobile_frac(d, "pre", agg, "speed", IMMOBILE_THR_MM_S))
        C["af_post"].append(band_immobile_frac(d, "post", agg, "speed", IMMOBILE_THR_MM_S))
        C["if_pre"].append(band_immobile_frac(d, "pre", inert, "speed", IMMOBILE_THR_MM_S))
        C["if_post"].append(band_immobile_frac(d, "post", inert, "speed", IMMOBILE_THR_MM_S))
        for epoch, suf in (("pre", "pre"), ("post", "post")):
            du = band_epoch_durations(d, epoch, agg)
            C[f"dur_{suf}"].append(np.median(du) if du.size else np.nan)
            C[f"pk_{suf}"].append(bout_vigor(d, epoch, agg, "peak_speed_mm_s"))
            C[f"path_{suf}"].append(bout_vigor(d, epoch, agg, "path_length_mm"))
            C[f"bdur_{suf}"].append(bout_vigor(d, epoch, agg, "duration_s"))
    C = {k: np.array(v, float) for k, v in C.items()}

    def rep(name, pre, post):
        dd, p, nn = paired(pre, post)
        print(f"{name:46s} pre={np.nanmean(pre):.3f} post={np.nanmean(post):.3f} Δ={np.mean(dd):+.3f} "
              f"up={int((dd > 0).sum())}/{nn} p={p:.3f}")

    print(f"n={int(np.isfinite(C['af_pre']).sum())}  (threshold {IMMOBILE_THR_MM_S} mm/s, 7-18mm band)\n")
    print("=== CONTROL: same metric, RAW vs SMOOTHED speed (aggressive) ===")
    rep("Aggressive P(immobile) [RAW speed]", C["raw_pre"], C["raw_post"])
    rep("Aggressive P(immobile) [SMOOTHED speed]", C["af_pre"], C["af_post"])
    rep("Inert P(immobile) [SMOOTHED speed]", C["if_pre"], C["if_post"])
    print("\n=== immobile-epoch median duration (s), aggressive mid-band ===")
    rep("Median immobile-epoch duration", C["dur_pre"], C["dur_post"])
    print("\n=== per-bout VIGOR in aggressive mid-band (median) ===")
    rep("Peak speed (mm/s)", C["pk_pre"], C["pk_post"])
    rep("Path length (mm)", C["path_pre"], C["path_post"])
    rep("Bout duration (s)", C["bdur_pre"], C["bdur_post"])


def run_sweep(loaded) -> None:
    D, sm_zero, sm_small = [], [], []
    for d in loaded:
        if "pre" not in d["epochs"] or "post" not in d["epochs"]:
            continue
        agg = d["roles"]["aggressive"]
        D.append((d, agg))
        s0, e0 = d["epochs"]["pre"]
        sp = d["speed"][s0:e0 + 1]
        v = np.isfinite(sp)
        sm_zero.append(float((sp[v] == 0).mean()))
        sm_small.append(float(((sp[v] > 0) & (sp[v] < 1)).mean()))
    print(f"\nn={len(D)}  smoothed speed: frac exactly 0 = {np.mean(sm_zero):.2f}, "
          f"frac in (0,1) mm/s = {np.mean(sm_small):.2f}\n")
    print(f"{'thr(mm/s)':>9} | {'RAW Δ':>7} {'p':>6} | {'SMOOTH Δ':>8} {'p':>6}")
    for thr in SWEEP_THRESHOLDS:
        rp = np.array([band_immobile_frac(d, "pre", a, "speed_raw", thr) for d, a in D])
        ro = np.array([band_immobile_frac(d, "post", a, "speed_raw", thr) for d, a in D])
        smp = np.array([band_immobile_frac(d, "pre", a, "speed", thr) for d, a in D])
        smo = np.array([band_immobile_frac(d, "post", a, "speed", thr) for d, a in D])

        def dp(pre, post):
            mm = np.isfinite(pre) & np.isfinite(post)
            dd = post[mm] - pre[mm]
            return np.mean(dd), wilcoxon_signed_rank_p_value(dd)[0]

        rd, rpp = dp(rp, ro)
        sd, spp = dp(smp, smo)
        print(f"{thr:9.1f} | {rd:+7.3f} {rpp:6.3f} | {sd:+8.3f} {spp:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-only", action="store_true", help="Only the threshold sweep.")
    ap.add_argument("--recompute-only", action="store_true", help="Only the raw-vs-smoothed recompute.")
    args = parse_standalone_exploratory_args(
        ap,
        analysis_id="goodcopbadcop_immobility_artifact",
    )

    loaded = []
    for rid, zp in resolve_cohort():
        try:
            loaded.append(load(zp))
        except ChaserDistanceReadError:
            raise
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], type(ex).__name__, ex)
    if not args.sweep_only:
        run_recompute(loaded)
    if not args.recompute_only:
        run_sweep(loaded)


if __name__ == "__main__":
    main()
