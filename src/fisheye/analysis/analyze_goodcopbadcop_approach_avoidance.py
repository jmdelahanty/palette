#!/usr/bin/env python
"""Reproducible: GoodCopBadCop approach-avoidance direction metrics (a documented null).

Direction-of-motion metrics in the 7-18 mm band, pre->post, aggressive vs inert. These
are robust to the speed noise floor because they are direction, not magnitude:
  fish_radial_velocity_moving  (+ toward object, - away)
  fraction_moving_away         (higher = more avoidance)
  approach_fraction  (scalar)  (lower = more avoidance)
  min_distance_mm    (scalar)  (higher = doesn't get as close)

Run (palette env):
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_approach_avoidance --exploratory-only

Kept as a durable NEGATIVE control: no learned directional avoidance survives (radial
velocity no change; fraction-moving-away decreased via selection bias; min-distance only
marginal and non-specific). Reinforces that this is an acute-threat-response dataset, not
a learned-spatial-avoidance one. Uses the chaser_response_regimes result (now on the
smoothed signal, sun 7e931481).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.chaser_response_regimes import build_chaser_response_regimes_result as build
from fisheye.analysis.cra_primary_endpoint import resolve_object_roles_from_protocol_payload
from fisheye.analysis.goodcopbadcop_common import (
    parse_standalone_exploratory_args,
    resolve_cohort,
    role_index,
    role_name,
)
from fisheye.group_statistics.paired import paired_sign_flip_p_value, wilcoxon_signed_rank_p_value

MID_BAND_MM = (7.0, 18.0)
MIN_WEIGHT = 30


def roles(res):
    r = zarr.open_group(res.zarr_path, mode="r")
    stim = r[res.source_stimulus_path]
    rr = resolve_object_roles_from_protocol_payload(json.loads(str(stim.attrs["protocol_json"])))
    return (next(role_index(o) for o in rr if role_name(o) == "aggressive"),
            next(role_index(o) for o in rr if role_name(o) == "inert"))


def epoch_map(res):
    return {e.label.split("_")[0]: i for i, e in enumerate(res.epochs)}


def band_weighted(res, ei, ch, arr, wt) -> float:
    ctr = res.distance_bin_centers_mm
    sel = (ctr >= MID_BAND_MM[0]) & (ctr <= MID_BAND_MM[1])
    a = arr[ei, ch, sel]
    w = wt[ei, ch, sel].astype(float)
    ok = np.isfinite(a) & (w > 0)
    return float(np.average(a[ok], weights=w[ok])) if w[ok].sum() >= MIN_WEIGHT else np.nan


def main() -> None:
    parse_standalone_exploratory_args(
        argparse.ArgumentParser(description=__doc__),
        analysis_id="goodcopbadcop_approach_avoidance",
    )
    metrics = {k: {r: {e: [] for e in ("pre", "post")} for r in ("agg", "inert")}
               for k in ("radvel_moving", "frac_away", "approach", "min_dist")}
    n = 0
    for rid, zp in resolve_cohort():
        try:
            res = build(Path(zp))
            agg, inert = roles(res)
            em = epoch_map(res)
            ip, io = em["pre"], em["post"]
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        n += 1
        for role, ch in (("agg", agg), ("inert", inert)):
            for ep, ei in (("pre", ip), ("post", io)):
                metrics["radvel_moving"][role][ep].append(
                    band_weighted(res, ei, ch, res.fish_radial_velocity_moving_mm_s, res.moving_frame_count))
                metrics["frac_away"][role][ep].append(
                    band_weighted(res, ei, ch, res.fraction_moving_away, res.frame_count))
                metrics["approach"][role][ep].append(float(res.approach_fraction[ei, ch]))
                metrics["min_dist"][role][ep].append(float(res.min_distance_mm[ei, ch]))
    print(f"n = {n} recordings  (7-18mm band, direction metrics on smoothed 'moving' gate)\n")

    def report(label, role, key, avoid_dir):
        pre = np.array(metrics[key][role]["pre"], float)
        post = np.array(metrics[key][role]["post"], float)
        m = np.isfinite(pre) & np.isfinite(post)
        d = post[m] - pre[m]
        if d.size < 3:
            print(f"{label:40s} n={d.size} too few")
            return
        p = wilcoxon_signed_rank_p_value(d)[0]
        toward_avoid = int((d * avoid_dir > 0).sum())
        print(f"{label:40s} pre={np.mean(pre[m]):+7.3f} post={np.mean(post[m]):+7.3f} "
              f"Δ={np.mean(d):+.3f}  toward-avoidance {toward_avoid}/{d.size}  p={p:.3f}")

    print("=== fish_radial_velocity_moving (mm/s; + toward, - away; avoidance = MORE negative) ===")
    report("Aggressive", "agg", "radvel_moving", -1)
    report("Inert", "inert", "radvel_moving", -1)
    print("\n=== fraction_moving_away (avoidance = HIGHER) ===")
    report("Aggressive", "agg", "frac_away", +1)
    report("Inert", "inert", "frac_away", +1)
    print("\n=== approach_fraction (P toward | moving; avoidance = LOWER) ===")
    report("Aggressive", "agg", "approach", -1)
    report("Inert", "inert", "approach", -1)
    print("\n=== min_distance_mm (closest approach; avoidance = HIGHER) ===")
    report("Aggressive", "agg", "min_dist", +1)
    report("Inert", "inert", "min_dist", +1)

    def diff_in_diff(key, sign):
        da = np.array(metrics[key]["agg"]["post"], float) - np.array(metrics[key]["agg"]["pre"], float)
        di = np.array(metrics[key]["inert"]["post"], float) - np.array(metrics[key]["inert"]["pre"], float)
        m = np.isfinite(da) & np.isfinite(di)
        dd = (da[m] - di[m]) * sign
        if m.sum() >= 3:
            p = paired_sign_flip_p_value(dd, iterations=20000, rng=np.random.default_rng(0))[0]
            print(f"\nDiff-in-diff {key} (agg-inert, avoidance-signed): n={int(m.sum())} mean={np.mean(dd):+.3f} "
                  f"toward-avoidance {int((dd > 0).sum())}/{int(m.sum())} p={p:.3f}")

    diff_in_diff("frac_away", +1)
    diff_in_diff("radvel_moving", -1)


if __name__ == "__main__":
    main()
