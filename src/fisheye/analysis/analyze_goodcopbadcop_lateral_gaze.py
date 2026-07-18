#!/usr/bin/env python
"""Reproducible: does the fish keep the aggressive object LATERAL, and does it develop?

Tests the hypothesis that after training the fish holds the chaser off to its side
(lateral bearing) rather than in front/behind, and that this grows pre->post. The trap
is wall-following: if the fish wall-follows and the object sits near the wall, lateral
geometry comes "for free". So every laterality number is contrasted against the object's
ROTATED VIRTUAL TWINS (the object position rotated 60-300 deg about the arena centre --
identical wall proximity, no object), exactly as the chaser components do elsewhere.

Bearings are reconstructed per frame (validated to 0.0 deg against the stored
egocentric_bearing per_chaser bearing):
    bearing = wrap(-atan2(obj_y - fish_y, obj_x - fish_x) * 180/pi - fish_heading_deg)
using fish_centroid_arena_xy, chaser_arena_xy, and the stored fish_heading_deg. Virtual
bearings use the same formula on the object rotated about the arena centre. Sectors match
the component convention: front |b|<45, lateral 45<=|b|<=135, behind |b|>135.

Aggregation is FISH-LEVEL (fish = the unit): each fish contributes one object and one
mean-virtual lateral fraction per epoch. Tests: (1) does object laterality rise pre->post
(development)? (2) is post laterality object-specific (object vs virtual > 0) or just
wall-following? (3) does the object-minus-virtual excess itself develop pre->post?

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_lateral_gaze                # epoch-level
    python -m fisheye.analysis.analyze_goodcopbadcop_lateral_gaze --by-distance  # distance-resolved

`--by-distance` profiles front/lateral/behind vs distance to the object, object vs virtual,
per epoch. Key finding (n=33): object-specific lateral-keeping (excess over the virtual
control) is confined to the near shell (~0-16 mm) -- strongest at contact (0-8 mm) during
the chase (+0.15), and in the 8-16 mm shell pre/post (+0.08/+0.10, innate + retained, not
grown); beyond ~25 mm orientation is wall geometry (excess ~0). Same near-shell locus as
the avoidance steering and eye lock-on. Reads the canonical registry (goodcopbadcop_common).
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
    load_epochs,
    nav,
    resolve_cohort,
    resolve_object_roles,
)
from fisheye.shared.arena_geometry import resolve_arena_geometry
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

EPOCHS = ("pre", "chase", "post")
VIRTUAL_ROT_DEG = (60.0, 120.0, 180.0, 240.0, 300.0)
LATERAL = (45.0, 135.0)   # |bearing| in this band = object off to the side
BEARING_EDGES = np.arange(-180.0, 180.001, 15.0)
BEARING_CTR = 0.5 * (BEARING_EDGES[:-1] + BEARING_EDGES[1:])
DIST_EDGES = (0.0, 8.0, 16.0, 25.0, 40.0, 60.0)
DIST_LABELS = ("0-8", "8-16", "16-25", "25-40", "40-60")
DIST_CTR = np.array([0.5 * (DIST_EDGES[i] + DIST_EDGES[i + 1]) for i in range(len(DIST_LABELS))])
OBJECT_C = "#c1435b"
VIRTUAL_C = "#8a8a8a"
FRONT_C = "#3a7ca5"
BEHIND_C = "#8a8a8a"
EPOCH_C = {"pre": "#8a8a8a", "chase": "#c1435b", "post": "#1b3a6b"}


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def bearing_of(obj_xy, fish_xy, heading_deg):
    dx = obj_xy[:, 0] - fish_xy[:, 0]
    dy = obj_xy[:, 1] - fish_xy[:, 1]
    return wrap180(-np.degrees(np.arctan2(dy, dx)) - heading_deg)


def rotate_about(pt_xy, cx, cy, deg):
    th = np.radians(deg)
    x = pt_xy[:, 0] - cx
    y = pt_xy[:, 1] - cy
    return np.column_stack([cx + x * np.cos(th) - y * np.sin(th),
                            cy + x * np.sin(th) + y * np.cos(th)])


def load(zp: str):
    r = zarr.open_group(zp, mode="r")
    cd = latest(nav(r, ["analysis", "chaser_distance_runs"]))
    eb = latest(nav(cd, ["egocentric_bearing"]))
    roles = resolve_object_roles(r)
    agg = roles["aggressive"]
    fish = np.asarray(cd["positions"]["fish_centroid_arena_xy"][:], float)
    fvalid = np.asarray(cd["positions"]["fish_valid"][:], bool)
    chas = np.asarray(cd["positions"]["chaser_arena_xy"][:], float)
    cvalid = np.asarray(cd["positions"]["chaser_valid"][:], bool)[:, agg]
    obj = chas[:, agg, :]
    heading = np.asarray(eb["frames"]["fish_heading_deg"][:], float)
    hvalid = np.asarray(eb["frames"]["fish_heading_valid"][:], bool)
    ppm = float(cd.attrs.get("pixels_per_mm_projector"))
    geo, _ = resolve_arena_geometry(r, cd, pixels_per_mm=ppm)
    cx, cy = geo.center_x_px, geo.center_y_px
    if cx is None:
        raise ValueError("no arena geometry")
    base_valid = fvalid & cvalid & hvalid & np.isfinite(heading)

    refs = {"object": (obj, base_valid)}
    for rot in VIRTUAL_ROT_DEG:
        obj_r = rotate_about(obj, cx, cy, rot)
        refs[f"virt_{rot:.0f}"] = (obj_r, fvalid & hvalid & np.isfinite(heading))

    bearings, dists = {}, {}
    for key, (pos, val) in refs.items():
        bearings[key] = bearing_of(pos, fish, heading)
        dists[key] = np.hypot(pos[:, 0] - fish[:, 0], pos[:, 1] - fish[:, 1]) / ppm
    return {"bearings": bearings, "dists": dists,
            "valid": {k: v for k, (_, v) in refs.items()},
            "epochs": load_epochs(r)}


def sector_fractions(bearing, valid, dist, epoch_range, near_mm):
    s, e = epoch_range
    m = valid.copy()
    m[:s] = False
    m[e + 1:] = False
    m &= np.isfinite(bearing)
    ab = np.abs(bearing)
    def frac(sel):
        n = sel.sum()
        if n < 100:
            return np.nan, np.nan, np.nan, int(n)
        lat = np.mean((ab[sel] >= LATERAL[0]) & (ab[sel] <= LATERAL[1]))
        front = np.mean(ab[sel] < LATERAL[0])
        behind = np.mean(ab[sel] > LATERAL[1])
        return float(lat), float(front), float(behind), int(n)
    all_f = frac(m)
    near_f = frac(m & (dist < near_mm)) if near_mm is not None else all_f
    return all_f, near_f


def sector_in_range(bearing, valid, dist, epoch_range, dlo, dhi, min_frames=80):
    """Front/lateral/behind fractions restricted to one epoch and one distance band."""
    s, e = epoch_range
    m = valid.copy()
    m[:s] = False
    m[e + 1:] = False
    m &= np.isfinite(bearing) & (dist >= dlo) & (dist < dhi)
    if m.sum() < min_frames:
        return (np.nan, np.nan, np.nan, int(m.sum()))
    ab = np.abs(bearing[m])
    return (float(np.mean(ab < LATERAL[0])),
            float(np.mean((ab >= LATERAL[0]) & (ab <= LATERAL[1]))),
            float(np.mean(ab > LATERAL[1])), int(m.sum()))


def bearing_hist(bearing, valid, epoch_range):
    s, e = epoch_range
    m = valid.copy(); m[:s] = False; m[e + 1:] = False; m &= np.isfinite(bearing)
    h, _ = np.histogram(bearing[m], bins=BEARING_EDGES, density=True)
    return h


def emit_by_distance(bydist, n, args, out_dir):
    """Distance-resolved orientation figure + tests (front/lateral/behind vs distance)."""
    def A(e, db, key):
        return np.array(bydist[e][db][key], float)

    def msem(a):
        a = a[np.isfinite(a)]
        return (np.nan, np.nan) if a.size < 3 else (float(np.mean(a)), float(np.std(a) / np.sqrt(a.size)))

    def excess(e, db):
        return A(e, db, "o_lat") - A(e, db, "v_lat")

    x = DIST_CTR
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True)
    for ei, e in enumerate(EPOCHS):
        ax = axes[0, ei]
        for key, col, lw, lab in (("o_front", FRONT_C, 1.5, "front |b|<45"),
                                  ("o_lat", OBJECT_C, 2.2, "lateral 45-135"),
                                  ("o_beh", BEHIND_C, 1.5, "behind |b|>135")):
            m = [msem(A(e, db, key))[0] for db in DIST_LABELS]
            s = [msem(A(e, db, key))[1] for db in DIST_LABELS]
            ax.errorbar(x, m, yerr=s, fmt="-o", color=col, lw=lw, ms=4, capsize=2, label=lab)
        vl = [msem(A(e, db, "v_lat"))[0] for db in DIST_LABELS]
        ax.plot(x, vl, "--", color=VIRTUAL_C, lw=1.4, label="virtual lateral")
        ax.set_title(e, fontsize=12, weight="bold")
        ax.set_ylim(0, 0.8)
        if ei == 0:
            ax.set_ylabel("bearing-sector fraction (object)")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax = axes[1, ei]
        exc = [msem(excess(e, db))[0] for db in DIST_LABELS]
        es = [msem(excess(e, db))[1] for db in DIST_LABELS]
        ax.axhline(0, color="#bbbbbb", lw=1)
        ax.errorbar(x, exc, yerr=es, fmt="-o", color=EPOCH_C[e], lw=2, ms=4, capsize=2)
        for di, db in enumerate(DIST_LABELS):
            d = excess(e, db); d = d[np.isfinite(d)]
            if d.size >= 6 and wilcoxon_signed_rank_p_value(d)[0] < 0.05:
                ax.plot(x[di], exc[di], "o", color=EPOCH_C[e], ms=10, mfc="none", mew=1.9)
        ax.set_ylim(-0.10, 0.22)
        ax.set_xlabel("distance to object (mm)")
        if ei == 0:
            ax.set_ylabel("lateral excess (object − virtual)")
    fig.suptitle(f"Orientation vs distance to the aggressive object — object vs virtual control (n={n})",
                 fontsize=13, weight="bold", y=1.01)
    fig.text(0.5, -0.01, "Open rings = object lateral fraction exceeds the wall-matched virtual control "
             "(Wilcoxon p<0.05, fish = unit). Object-specific orienting is confined to the near shell.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_lateral_gaze_by_distance_{args.tag}.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, f" (n_fish={n})\n")

    # ---- text report ----
    print("Orientation of the aggressive object vs distance (cohort mean; front|b|<45, lateral 45-135, behind>135)\n")
    for e in EPOCHS:
        print(f"[{e}]   dist(mm):  " + "  ".join(f"{db:>7s}" for db in DIST_LABELS))
        for lab, key in (("front  ", "o_front"), ("lateral", "o_lat"), ("behind ", "o_beh")):
            print(f"   {lab}      " + "  ".join(f"{msem(A(e, db, key))[0]:7.2f}" for db in DIST_LABELS))
        print(f"   L-exc/vir   " + "  ".join(f"{msem(excess(e, db))[0]:+7.2f}" for db in DIST_LABELS))
        specific = [db for db in DIST_LABELS
                    if (excess(e, db)[np.isfinite(excess(e, db))].size >= 6
                        and wilcoxon_signed_rank_p_value(excess(e, db)[np.isfinite(excess(e, db))])[0] < 0.05
                        and np.nanmean(excess(e, db)) > 0)]
        print(f"   object-specific lateral (excess>0, p<0.05) at: {', '.join(specific) or '(none)'} mm\n")

    print("Development pre->post (paired, fish-level), per distance bin:")
    for db in DIST_LABELS:
        op, oo = A("pre", db, "o_lat"), A("post", db, "o_lat")
        m = np.isfinite(op) & np.isfinite(oo)
        dl = oo[m] - op[m]
        xp, xo = excess("pre", db), excess("post", db)
        me = np.isfinite(xp) & np.isfinite(xo)
        dx = xo[me] - xp[me]
        pl = wilcoxon_signed_rank_p_value(dl)[0] if dl.size >= 6 else np.nan
        px = wilcoxon_signed_rank_p_value(dx)[0] if dx.size >= 6 else np.nan
        print(f"  {db:>6s} mm: object lateral Δ={np.mean(dl) if dl.size else np.nan:+.3f} p={pl:.3f} (n={dl.size}); "
              f"excess Δ={np.mean(dx) if dx.size else np.nan:+.3f} p={px:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--near-mm", type=float, default=25.0, help="Near-distance conditioning (mm).")
    ap.add_argument("--by-distance", action="store_true",
                    help="Profile orientation as a function of distance to the object (distance-resolved figure + tests).")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-18")
    args = ap.parse_args()
    out_dir = args.out_dir or figures_dir()

    # per fish: object & mean-virtual lateral fraction per epoch (all + near), + bearing hists
    lat = {sc: {"object": {e: [] for e in EPOCHS}, "virtual": {e: [] for e in EPOCHS}}
           for sc in ("all", "near")}
    hists = {"object": {e: [] for e in EPOCHS}, "virtual": {e: [] for e in EPOCHS}}
    # per fish: front/lateral/behind (object) + lateral (virtual) per epoch per distance bin
    bydist = {e: {db: {"o_front": [], "o_lat": [], "o_beh": [], "v_lat": []} for db in DIST_LABELS} for e in EPOCHS}
    n = 0
    for rid, zp in resolve_cohort():
        try:
            d = load(zp)
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], type(ex).__name__, ex)
            continue
        if not all(e in d["epochs"] for e in EPOCHS):
            continue
        n += 1
        virt_keys = [k for k in d["bearings"] if k.startswith("virt_")]
        for e in EPOCHS:
            rng = d["epochs"][e]
            oa, on = sector_fractions(d["bearings"]["object"], d["valid"]["object"], d["dists"]["object"], rng, args.near_mm)
            lat["all"]["object"][e].append(oa[0]); lat["near"]["object"][e].append(on[0])
            v_all, v_near = [], []
            for vk in virt_keys:
                va, vn = sector_fractions(d["bearings"][vk], d["valid"][vk], d["dists"][vk], rng, args.near_mm)
                v_all.append(va[0]); v_near.append(vn[0])
            lat["all"]["virtual"][e].append(np.nanmean(v_all)); lat["near"]["virtual"][e].append(np.nanmean(v_near))
            hists["object"][e].append(bearing_hist(d["bearings"]["object"], d["valid"]["object"], rng))
            hists["virtual"][e].append(np.nanmean([bearing_hist(d["bearings"][vk], d["valid"][vk], rng) for vk in virt_keys], axis=0))
            for di, db in enumerate(DIST_LABELS):
                lo, hi = DIST_EDGES[di], DIST_EDGES[di + 1]
                of, ol, ob, _ = sector_in_range(d["bearings"]["object"], d["valid"]["object"], d["dists"]["object"], rng, lo, hi)
                vl = np.nanmean([sector_in_range(d["bearings"][vk], d["valid"][vk], d["dists"][vk], rng, lo, hi)[1] for vk in virt_keys])
                bydist[e][db]["o_front"].append(of); bydist[e][db]["o_lat"].append(ol)
                bydist[e][db]["o_beh"].append(ob); bydist[e][db]["v_lat"].append(vl)

    if args.by_distance:
        emit_by_distance(bydist, n, args, out_dir)
        return

    def arr(sc, role, e):
        return np.array(lat[sc][role][e], float)

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ei, e in enumerate(EPOCHS):
        ax = axes[0, ei]
        mo = np.nanmean(hists["object"][e], axis=0); mv = np.nanmean(hists["virtual"][e], axis=0)
        ax.plot(BEARING_CTR, mo, "-", color=OBJECT_C, lw=2.2, label="aggressive object")
        ax.plot(BEARING_CTR, mv, "--", color=VIRTUAL_C, lw=1.6, label="virtual control")
        for b in (-135, -45, 45, 135):
            ax.axvline(b, color="#dddddd", lw=1, zorder=0)
        ax.axvspan(45, 135, color="#f2c94c", alpha=0.10, zorder=0)
        ax.axvspan(-135, -45, color="#f2c94c", alpha=0.10, zorder=0)
        ax.set_title(f"{e}  (bearing density)", fontsize=11, weight="bold")
        ax.set_xlabel("object bearing (deg)  0=front, ±90=lateral, ±180=behind")
        ax.set_xticks([-180, -90, 0, 90, 180])
        if ei == 0:
            ax.set_ylabel("frame density"); ax.legend(frameon=False, fontsize=8)
    # bottom: lateral fraction object vs virtual, all + near; excess vs epoch
    x = np.arange(len(EPOCHS))
    for ci, sc in enumerate(("all", "near")):
        ax = axes[1, ci]
        mo = [np.nanmean(arr(sc, "object", e)) for e in EPOCHS]
        so = [np.nanstd(arr(sc, "object", e)) / np.sqrt(np.isfinite(arr(sc, "object", e)).sum()) for e in EPOCHS]
        mv = [np.nanmean(arr(sc, "virtual", e)) for e in EPOCHS]
        sv = [np.nanstd(arr(sc, "virtual", e)) / np.sqrt(np.isfinite(arr(sc, "virtual", e)).sum()) for e in EPOCHS]
        ax.errorbar(x - 0.05, mo, yerr=so, fmt="-o", color=OBJECT_C, lw=2, capsize=3, label="object")
        ax.errorbar(x + 0.05, mv, yerr=sv, fmt="--s", color=VIRTUAL_C, lw=1.6, capsize=3, label="virtual")
        ax.set_xticks(x); ax.set_xticklabels(EPOCHS)
        ax.set_title(f"Lateral fraction ({sc}{'' if sc=='all' else f' <{args.near_mm:.0f}mm'})", fontsize=11, weight="bold")
        ax.set_ylabel("P(object lateral, 45-135°)")
        if ci == 0:
            ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 2]
    for sc, col in (("all", "#1b3a6b"), ("near", "#c1435b")):
        exc = [np.nanmean(arr(sc, "object", e) - arr(sc, "virtual", e)) for e in EPOCHS]
        se = [np.nanstd(arr(sc, "object", e) - arr(sc, "virtual", e)) / np.sqrt(np.isfinite(arr(sc, "object", e) - arr(sc, "virtual", e)).sum()) for e in EPOCHS]
        ax.errorbar(x, exc, yerr=se, fmt="-o", color=col, lw=2, capsize=3, label=sc)
    ax.axhline(0, color="#bbbbbb", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(EPOCHS)
    ax.set_title("Object-specific laterality (object − virtual)", fontsize=11, weight="bold")
    ax.set_ylabel("lateral fraction excess"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Does the fish keep the aggressive object lateral? Object vs virtual control (n={n})",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_lateral_gaze_{args.tag}.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, f" (n_fish={n})\n")

    # ---- text report ----
    def paired(a, b):
        m = np.isfinite(a) & np.isfinite(b); d = b[m] - a[m]
        return (np.mean(d), wilcoxon_signed_rank_p_value(d)[0], int(m.sum())) if m.sum() >= 6 else (np.nan, np.nan, int(m.sum()))
    def onesample(a):
        a = a[np.isfinite(a)]
        return (np.mean(a), wilcoxon_signed_rank_p_value(a)[0], a.size) if a.size >= 6 else (np.nan, np.nan, a.size)

    for sc in ("all", "near"):
        print(f"=== lateral fraction ({sc}{'' if sc=='all' else f', <{args.near_mm:.0f}mm'}) ===")
        for e in EPOCHS:
            print(f"  {e:6s} object={np.nanmean(arr(sc,'object',e)):.3f}  virtual={np.nanmean(arr(sc,'virtual',e)):.3f}  "
                  f"excess={np.nanmean(arr(sc,'object',e)-arr(sc,'virtual',e)):+.3f}")
        dm, dp, dn = paired(arr(sc, "object", "pre"), arr(sc, "object", "post"))
        print(f"  (1) DEVELOPMENT  object laterality pre->post: Δ={dm:+.3f} p={dp:.3f} n={dn}")
        em, ep, en = onesample(arr(sc, "object", "post") - arr(sc, "virtual", "post"))
        print(f"  (2) OBJECT-SPECIFIC post (object-virtual vs 0): mean={em:+.3f} p={ep:.3f} n={en}  "
              f"{'object-specific' if (np.isfinite(ep) and ep<0.05 and em>0) else 'wall-following (n.s.)'}")
        xm, xp, xn = paired(arr(sc, "object", "pre") - arr(sc, "virtual", "pre"),
                            arr(sc, "object", "post") - arr(sc, "virtual", "post"))
        print(f"  (3) EXCESS develops pre->post: Δ={xm:+.3f} p={xp:.3f} n={xn}\n")


if __name__ == "__main__":
    main()
