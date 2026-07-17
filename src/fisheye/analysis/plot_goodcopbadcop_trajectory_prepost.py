#!/usr/bin/env python
"""Reproducible pre/post trajectory + immobility figure for one GoodCopBadCop recording.

Shows the fish path (grey) and where it holds still (dark points) relative to the
aggressive vs inert object, before and after chase training. Immobility uses the
SMOOTHED signal (speed_smoothed_mm < 1 mm/s, ≈ not in a bout) -- the raw centroid-speed
version was a tracking-noise artifact (see chaser_response_regimes contract).
Illustrative single recording only: neither the occupancy void nor mid-band immobility
reaches cohort significance. The solid GoodCopBadCop result is the escape response.

Run (palette env):
    ~/miniconda3/envs/palette-py311/bin/python -m fisheye.analysis.plot_goodcopbadcop_trajectory_prepost

The figure is written OUTSIDE the repo (to $PALETTE_RECORDINGS_ROOT/figures); this script is
committed but its output is not.

Options:
    --recording-id  substring of the recording (default: the clean 0%-dropout demo,
                    2026-06-14T21-50-10Z_arena_3). Resolved to its analysis zarr via
                    the registry, so it survives path moves.
    --out           output PNG path (default: $PALETTE_RECORDINGS_ROOT/figures/
                    goodcopbadcop_trajectory_prepost_<rid>.png).

Data source: refined-detection fish centroid (offline) + live-logged chaser position,
from <rec>_analysis.zarr. Immobility uses centroid speed < 1 mm/s on validly-tracked
frame pairs -- a locomotor proxy, not ethological freezing. y-axis is inverted because
arena coordinates are image-down.

Provenance for the figures this produced: see
/nvme1/recordings/figures/goodcopbadcop_freeze_figures_2026-07-17_PROVENANCE.txt
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fisheye.analysis.chaser_response_regimes import (
    build_chaser_response_regimes_result as build,
    _load_smoothed_immobility_speed,
)
from fisheye.analysis.cra_primary_endpoint import resolve_object_roles_from_protocol_payload

REGISTRY = "/nvme1/palette_registry.sqlite"
# Figures live OUTSIDE the repo (this script is committed; its output is not).
FIGURES_DIR = Path(os.environ.get("PALETTE_RECORDINGS_ROOT", "/nvme1/recordings")) / "figures"
DEFAULT_RID = "2026-06-14T21-50-10Z_arena_3"        # clean 0%-dropout demo recording
AGG_C = "#c1435b"; INERT_C = "#3a7ca5"; IMMOB_C = "#6a1b3d"


def resolve_zarr(recording_like: str) -> tuple[str, str]:
    """Return (recording_id, analysis_zarr_path) from the registry (path-move safe)."""
    conn = sqlite3.connect(REGISTRY); conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT recording_id, zarr_path FROM dataset_context_current "
            "WHERE recording_id LIKE ? AND zarr_use='analysis' AND dataset_status='active' "
            "ORDER BY recording_id LIMIT 1",
            (f"%{recording_like}%",),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not Path(row["zarr_path"]).is_dir():
        raise SystemExit(f"No reachable active analysis zarr for recording like {recording_like!r}.")
    return row["recording_id"], row["zarr_path"]


def immobile_mask(smoothed_speed: np.ndarray, valid: np.ndarray, sl: slice):
    """Immobile = smoothed speed < 1 mm/s. speed_smoothed_mm is deadbanded between
    bouts, so this reads as "not in a bout" -- NOT the raw centroid diff, whose
    ~1.6 mm/s jitter floor made the raw immobility metric a tracking-noise artifact."""
    sp = smoothed_speed[sl]; tracked = valid[sl] & np.isfinite(sp)
    return tracked, tracked & (sp < 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording-id", default=DEFAULT_RID)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rid, zpath = resolve_zarr(args.recording_id)
    out = args.out or FIGURES_DIR / f"goodcopbadcop_trajectory_prepost_{rid.split('_GoodCop')[0]}.png"
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})

    res = build(Path(zpath))
    r = zarr.open_group(zpath, mode="r")
    pos = r[f"analysis/chaser_distance_runs/{res.chaser_distance_run_name}"]["positions"]
    ppm = res.pixels_per_mm_projector
    fish = np.asarray(pos["fish_centroid_arena_xy"][:], float) / ppm    # (N,2) mm
    chas = np.asarray(pos["chaser_arena_xy"][:], float) / ppm           # (N,nch,2) mm
    fvalid = np.asarray(pos["fish_valid"][:], bool)
    cvalid = np.asarray(pos["chaser_valid"][:], bool)
    # Role AND rendered colour both come from the experiment metadata (protocol_json ->
    # CRA behavior classes). Label by ROLE (counterbalance-proof); annotate the actual hex.
    role_by = {o.object_role: o for o in resolve_object_roles_from_protocol_payload(
        json.loads(str(r[res.source_stimulus_path].attrs["protocol_json"])))}
    agg_obj, inert_obj = role_by["aggressive"], role_by["inert"]
    agg, inr = agg_obj.object_index, inert_obj.object_index
    agg_hex = agg_obj.raw_color_hex or AGG_C
    inert_hex = inert_obj.raw_color_hex or INERT_C
    agg_lbl, inert_lbl = f"aggressive ({agg_hex})", f"inert ({inert_hex})"
    em = {e.label.split("_")[0]: e for e in res.epochs}
    fps = float(res.fps)
    smoothed_speed, _imm_src = _load_smoothed_immobility_speed(r, fish.shape[0])
    if smoothed_speed is None:
        raise SystemExit("track_kinematics speed_smoothed_mm unavailable; refusing to plot on raw "
                         "centroid speed (that immobility metric is a tracking-noise artifact).")

    def epoch_slice(lbl):
        e = em[lbl]; return slice(e.start_frame, e.end_frame + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharex=True, sharey=True)
    for ax, lbl, ttl in zip(axes, ["pre", "post"], ["Pre-training", "Post-training (after chase)"]):
        sl = epoch_slice(lbl)
        v, im = immobile_mask(smoothed_speed, fvalid, sl)
        f = fish[sl]; cv = cvalid[sl]

        def obj_xy(idx):
            c = chas[sl][:, idx, :]; ok = cv[:, idx] & np.isfinite(c).all(1)
            return np.median(c[ok], 0)

        ax_pos, in_pos = obj_xy(agg), obj_xy(inr)
        fv = f.copy(); fv[~v] = np.nan
        ax.plot(fv[:, 0], fv[:, 1], "-", color="#c9c9c9", lw=0.4, alpha=0.7, zorder=1)
        ax.scatter(f[im, 0], f[im, 1], s=6, color=IMMOB_C, alpha=0.35, edgecolors="none", zorder=2,
                   label="immobile (smoothed <1 mm/s ≈ not in a bout)")
        for rr in (7, 18):
            ax.add_patch(Circle(ax_pos, rr, fill=False, ec=agg_hex, ls="--", lw=1.0, alpha=0.6, zorder=3))
        ax.scatter(*ax_pos, s=260, color=agg_hex, edgecolors="black", lw=1.2, marker="o", zorder=5, label=agg_lbl)
        ax.scatter(*in_pos, s=200, color=inert_hex, edgecolors="black", lw=1.2, marker="s", zorder=5, label=inert_lbl)
        ax.set_title(ttl, fontsize=13, weight="bold")
        ax.set_aspect("equal"); ax.set_xlabel("x (mm)")
        ax.text(0.03, 0.97, f"immobile {100*im.sum()/max(v.sum(),1):.0f}% of tracked time",
                transform=ax.transAxes, va="top", fontsize=10, color=IMMOB_C)
    axes[0].set_ylabel("y (mm)")
    axes[0].invert_yaxis()  # arena y is image-down; sharey=True flips both panels once
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9, markerscale=0.8)
    fig.suptitle(f"Where the fish holds still, pre vs post — example recording {rid.split('_GoodCop')[0]}",
                 fontsize=13.5, weight="bold", y=1.0)
    fig.text(0.5, -0.035, "Dark = immobile positions; grey = path. Dashed rings = 7–18 mm shell around the aggressive object. "
             "Note the reduced occupancy around the aggressive object post-training (spatial avoidance).",
             ha="center", fontsize=8.5, color="#666")
    fig.text(0.5, -0.075, "Illustrative single recording. Immobility uses the smoothed signal (≈ not in a bout). "
             "Neither the occupancy void nor mid-band immobility reaches cohort significance; the solid "
             "GoodCopBadCop result is the escape response (12x during chase).",
             ha="center", fontsize=8.0, color="#999", style="italic")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=160)
    print(f"recording: {rid}\nzarr: {zpath}\nwrote {out}")
    for lbl in ("pre", "post"):
        v, im = immobile_mask(smoothed_speed, fvalid, epoch_slice(lbl))
        print(f"  {lbl}: tracked {int(v.sum())} frames, immobile {int(im.sum())} "
              f"({100*im.sum()/max(v.sum(),1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
