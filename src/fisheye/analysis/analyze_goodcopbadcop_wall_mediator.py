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
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_wall_mediator --exploratory-only

Key result (full canonical cohort): chase is heavily near-wall and angular R rises from
pre to chase (localized, NOT uniform thigmotaxis) -- the chaser drives the fish to a
wall sector. On the full cohort (n=28) the freeze-over-trials habituation SURVIVES even
the wall over-control: partial r(freeze, trial | wall) = +0.23, p=0.003 (raw r=+0.28,
p<0.001). At n=11 this partial was +0.11, p=0.47 (looked killed) -- an underpowering
artifact, not evidence against the habituation. Read wall as a mediator regardless.
See docs/diagnostics/goodcopbadcop_behavior_synthesis_handoff_2026-07-17.md.
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

OUTER_WALL_FRAC = 0.8   # positions with center-distance > 0.8 * radius = "near wall" (outer 20%)
LATE_FROM = 5


def r_of(angles: np.ndarray) -> float:
    return float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))) if angles.size else np.nan


def load_angular(zp: str):
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_arena_geometry_authority()


def load_partial(zp: str):
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_escape_freeze")


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
        except ChaserDistanceReadError:
            raise
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
        except ChaserDistanceReadError:
            raise
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
    args = parse_standalone_exploratory_args(
        ap,
        analysis_id="goodcopbadcop_wall_mediator",
    )
    cohort = resolve_cohort()
    if not args.partial_only:
        run_angular(cohort)
    if not args.angular_only:
        run_partial(cohort)


if __name__ == "__main__":
    main()
