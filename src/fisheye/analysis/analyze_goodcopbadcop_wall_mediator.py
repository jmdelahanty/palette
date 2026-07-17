#!/usr/bin/env python
"""Reproducible: GoodCopBadCop wall-proximity is a chase MEDIATOR, not uniform thigmotaxis.

Two analyses that together show why you must NOT partial wall-distance out of the
habituation:

  angular : angular concentration (mean resultant length R) of near-wall fish positions
            per epoch. R~0 = spread around the whole perimeter (true thigmotaxis);
            R~1 = concentrated in one sector (driven/parked). If the chase R >> pre R,
            the chaser is driving the fish to a *localized* wall sector, not inducing
            uniform wall-hugging.

  partial : per chase trial, median fish distance-from-arena-center (wall proximity) and
            freeze fraction. (1) does wall proximity rise over trials? (2) does freeze
            track wall proximity? (3) raw vs partial corr(freeze, trial | wall). The
            partial is shown to DEMONSTRATE the over-control trap, not to license it:
            because the chase causes the wall-proximity, controlling for wall removes
            part of the real chase effect.

Open dish, no physical trapping -- this is a position/mediation question, not "cornered".

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_wall_mediator

Key result (2026-07-17): chase ~91% near-wall, angular R rises pre ~0.30 -> chase ~0.55
(localized, NOT uniform thigmotaxis). freeze~trial|wall partial ~0.11 (p~0.47) is an
over-control; the raw freeze-over-trials signal should be read via mediation, not this
partial. See docs/diagnostics/goodcopbadcop_behavior_synthesis_handoff_2026-07-17.md.
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr

from fisheye.analysis.goodcopbadcop_common import load_epochs, open_distance_run, resolve_cohort
from fisheye.shared.arena_geometry import resolve_arena_geometry
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

OUTER_WALL_FRAC = 0.8   # positions with center-distance > 0.8 * radius = "near wall" (outer 20%)
LATE_FROM = 5


def _geometry(root, cd):
    ppm = float(cd.attrs.get("pixels_per_mm_projector"))
    geo, _ = resolve_arena_geometry(root, cd, pixels_per_mm=ppm)
    if geo.center_x_px is None:
        raise ValueError("no arena geometry")
    return geo, ppm


def r_of(angles: np.ndarray) -> float:
    return float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))) if angles.size else np.nan


def load_angular(zp: str):
    r = zarr.open_group(zp, mode="r")
    cd = open_distance_run(r)
    geo, _ = _geometry(r, cd)
    fish = np.asarray(cd["positions"]["fish_centroid_arena_xy"][:], float)
    fv = np.asarray(cd["positions"]["fish_valid"][:], bool)
    cx, cy, rpx = geo.center_x_px, geo.center_y_px, geo.radius_px
    cdist = np.hypot(fish[:, 0] - cx, fish[:, 1] - cy)
    ang = np.arctan2(fish[:, 1] - cy, fish[:, 0] - cx)
    out = {}
    for epoch, (s, e) in load_epochs(r).items():
        sl = slice(s, e + 1)
        finite = fv[sl] & np.isfinite(cdist[sl])
        near = finite & (cdist[sl] > OUTER_WALL_FRAC * rpx)
        frac_nearwall = near.sum() / max(finite.sum(), 1)
        out[epoch] = (r_of(ang[sl][near]), float(frac_nearwall), int(near.sum()))
    return out


def load_partial(zp: str):
    r = zarr.open_group(zp, mode="r")
    cd = open_distance_run(r)
    geo, ppm = _geometry(r, cd)
    fish = np.asarray(cd["positions"]["fish_centroid_arena_xy"][:], float)
    fv = np.asarray(cd["positions"]["fish_valid"][:], bool)
    cdist = np.hypot(fish[:, 0] - geo.center_x_px, fish[:, 1] - geo.center_y_px) / ppm  # mm from centre
    ef = cd["chaser_escape_freeze"][sorted(cd["chaser_escape_freeze"].group_keys())[-1]]
    ordv = np.asarray(ef["trials"]["trial_ordinal"][:], int)
    sf = np.asarray(ef["trials"]["start_frame"][:], int)
    en = np.asarray(ef["trials"]["end_frame"][:], int)
    ff = np.asarray(ef["trial_metrics"]["freeze_low_speed_fraction"][:], float)
    rows = []
    for o, s, e, f in zip(ordv, sf, en, ff):
        seg = cdist[s:e + 1]
        v = fv[s:e + 1] & np.isfinite(seg)
        if v.sum() >= 20 and np.isfinite(f):
            rows.append((int(o), float(np.median(seg[v])), float(f)))
    return np.array(rows)  # (n,3): ordinal, centre_dist_mm, freeze


def partial_r(y, x, z) -> float:
    """corr(y, x) controlling for z."""
    def rr(a, b):
        return np.corrcoef(a, b)[0, 1]
    ryx, ryz, rxz = rr(y, x), rr(y, z), rr(x, z)
    den = np.sqrt(max(1e-9, (1 - ryz ** 2) * (1 - rxz ** 2)))
    return (ryx - ryz * rxz) / den


def run_angular(cohort) -> None:
    rows = []
    for rid, zp in cohort:
        try:
            rows.append((rid.split("_GoodCop")[0], load_angular(zp)))
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
    print("R = angular concentration of near-wall positions (outer 20%). "
          "R~0 uniform (thigmotaxis), R~1 one sector.\n")
    for epoch in ("pre", "chase", "post"):
        rs = np.array([o[epoch][0] for _, o in rows if epoch in o and np.isfinite(o[epoch][0])])
        fw = np.array([o[epoch][1] for _, o in rows if epoch in o])
        if rs.size:
            print(f"{epoch:6s}: mean R = {np.nanmean(rs):.2f} (median {np.nanmedian(rs):.2f}, "
                  f"range {np.nanmin(rs):.2f}-{np.nanmax(rs):.2f})   near-wall time frac = {np.nanmean(fw):.2f}   n={rs.size}")
    print("\nper-recording chase R:")
    for rid, o in rows:
        if "chase" in o:
            print(f"  {rid:30s} R={o['chase'][0]:.2f}  near-wall frac={o['chase'][1]:.2f}")


def run_partial(cohort) -> None:
    r_freeze_ord, pr_freeze_ord, r_freeze_wall = [], [], []
    wall_early, wall_late = [], []
    n = 0
    for rid, zp in cohort:
        try:
            tr = load_partial(zp)
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        if tr.shape[0] < 5:
            continue
        n += 1
        o, wall, frz = tr[:, 0], tr[:, 1], tr[:, 2]  # larger centre-distance = nearer wall
        r_freeze_ord.append(np.corrcoef(frz, o)[0, 1])
        r_freeze_wall.append(np.corrcoef(frz, wall)[0, 1])
        pr_freeze_ord.append(partial_r(frz, o, wall))
        we = [wall[i] for i in range(len(o)) if o[i] <= 2]
        wl = [wall[i] for i in range(len(o)) if o[i] >= LATE_FROM]
        if we and wl:
            wall_early.append(np.mean(we))
            wall_late.append(np.mean(wl))
    r_freeze_ord = np.array(r_freeze_ord)
    pr = np.array(pr_freeze_ord)
    r_fw = np.array(r_freeze_wall)
    we, wl = np.array(wall_early), np.array(wall_late)
    print(f"\nn = {n} recordings.  centre-distance from arena centre (mm); larger = nearer wall.\n")
    print(f"1) Does the fish get NEARER the wall over trials? centre-dist early={we.mean():.1f} late={wl.mean():.1f} mm  "
          f"Δ={np.mean(wl - we):+.1f}  p={wilcoxon_signed_rank_p_value(wl - we)[0]:.3f}")
    print(f"2) Does freeze track wall proximity?  mean r(freeze, centre-dist) = {np.nanmean(r_fw):+.3f}  "
          f"(vs 0: p={wilcoxon_signed_rank_p_value(r_fw[np.isfinite(r_fw)])[0]:.3f})")
    print(f"3) Raw r(freeze, trial ordinal)      = {np.nanmean(r_freeze_ord):+.3f}  "
          f"(vs 0: p={wilcoxon_signed_rank_p_value(r_freeze_ord[np.isfinite(r_freeze_ord)])[0]:.3f})")
    print(f"   PARTIAL r(freeze, trial | wall)   = {np.nanmean(pr):+.3f}  "
          f"(vs 0: p={wilcoxon_signed_rank_p_value(pr[np.isfinite(pr)])[0]:.3f})   <- OVER-CONTROL (wall is a mediator)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angular-only", action="store_true")
    ap.add_argument("--partial-only", action="store_true")
    args = ap.parse_args()
    cohort = resolve_cohort()
    if not args.partial_only:
        run_angular(cohort)
    if not args.angular_only:
        run_partial(cohort)


if __name__ == "__main__":
    main()
