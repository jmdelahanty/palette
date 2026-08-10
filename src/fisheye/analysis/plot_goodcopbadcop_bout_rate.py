#!/usr/bin/env python
"""Reproducible: GoodCopBadCop bout rate + bout statistics per epoch (pre / chase / post).

Bout rate = valid bouts / validly-tracked minute (the denominator excludes tracking dropout).
Bouts come from `chaser_bout_response`; these are robust bout-level metrics, well above the
speed noise floor that made the frame-level immobility readout an artifact.

Modes:
  cohort (default): mean +/- SEM across all reachable recordings, faint per-recording lines,
      one recording highlighted (--example). Panels: bout rate, median bout duration, median
      inter-bout interval, median bout peak speed -- each pre / chase / post.
  single (--recording-id SUBSTR): one fish. Bout rate per epoch (bars) plus per-epoch
      distributions of bout duration, inter-bout interval, and peak speed.

Run (palette env):
    scripts/py -m fisheye.analysis.plot_goodcopbadcop_bout_rate --exploratory-only
    scripts/py -m fisheye.analysis.plot_goodcopbadcop_bout_rate --exploratory-only --recording-id 21-50-10Z_arena_3

Figures are written OUTSIDE the repo (to $PALETTE_RECORDINGS_ROOT/figures); this script is
committed but its output is not. Options: --recording-id (single-fish mode), --example
(highlight in cohort mode), --out-dir, --tag.

Key result (2026-07-17, n=12): bout rate collapses during the chase (~100 -> ~48/min,
pre->chase p<0.001, 12/12) and partly recovers post (~84/min) -- behavioural inhibition
under threat. Ties to the escape result: fewer TOTAL bouts during chase but 12x more escapes.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fisheye.analysis.goodcopbadcop_common import (
    parse_standalone_exploratory_args,
    resolve_cohort as cohort,
    save_standalone_exploratory_figure,
)
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

FIGURES_DIR = Path(os.environ.get("PALETTE_RECORDINGS_ROOT", "/nvme1/recordings")) / "figures"
EPOCHS = ["pre", "chase", "post"]
DEFAULT_EXAMPLE = "21-50-10Z_arena_3"
GRID = "#e6e6e6"; COHORT_C = "#1b3a6b"; EX_C = "#c1435b"
PANELS = [("rate", "Bouts per validly-tracked minute", "bouts/min"),
          ("dur", "Bout duration", "s"),
          ("ibi", "Inter-bout interval", "s"),
          ("pk", "Bout peak speed", "mm/s")]


# cohort() is the shared, registry-resolved, duplicate-deduped resolver (see goodcopbadcop_common).


def load(zp):
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_bout_response")


def epoch_bouts(d, epk):
    """Per-epoch bout rate + raw per-bout arrays (duration, IBI, peak)."""
    if epk not in d["ep"]:
        return dict(rate=np.nan, n=0, valid_min=np.nan, dur=np.array([]), ibi=np.array([]), pk=np.array([]))
    s, e = d["ep"][epk]; on = d["bs"]; m = (on >= s) & (on <= e)
    valid_min = d["fvalid"][s:e + 1].sum() / d["fps"] / 60.0
    onset = np.sort(on[m])
    ibi = np.diff(onset) / d["fps"] if onset.size > 1 else np.array([])
    return dict(rate=(m.sum() / valid_min if valid_min > 0.1 else np.nan), n=int(m.sum()), valid_min=valid_min,
                dur=d["bdur"][m], ibi=ibi, pk=d["bpk"][m])


def _median(x):
    return float(np.median(x)) if np.asarray(x).size else np.nan


# ------------------------------------------------------------------ cohort figure
def plot_cohort(recs, example_sub, out):
    C = {k: {ep: [] for ep in EPOCHS} for k in ("rate", "dur", "ibi", "pk")}
    example = None
    for rid, zp in recs:
        try:
            d = load(zp); row = {ep: epoch_bouts(d, ep) for ep in EPOCHS}
        except ChaserDistanceReadError:
            raise
        except Exception as ex:  # noqa: BLE001
            print("skip", rid.split("_GoodCop")[0], ex); continue
        vals = {ep: dict(rate=row[ep]["rate"], dur=_median(row[ep]["dur"]),
                         ibi=_median(row[ep]["ibi"]), pk=_median(row[ep]["pk"])) for ep in EPOCHS}
        for k in C:
            for ep in EPOCHS:
                C[k][ep].append(vals[ep][k])
        if example_sub and example_sub in rid:
            example = {k: [vals[ep][k] for ep in EPOCHS] for k in C}
    for k in C:
        for ep in EPOCHS:
            C[k][ep] = np.array(C[k][ep], float)
    n = len(C["rate"]["pre"])

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3)); x = np.arange(3)
    for ax, (key, ttl, unit) in zip(axes, PANELS):
        for i in range(n):
            ax.plot(x, [C[key][ep][i] for ep in EPOCHS], "-", color="#c9c9c9", lw=1, alpha=0.55, zorder=1)
        mean = [np.nanmean(C[key][ep]) for ep in EPOCHS]
        sem = [np.nanstd(C[key][ep]) / np.sqrt(np.sum(np.isfinite(C[key][ep]))) for ep in EPOCHS]
        ax.errorbar(x, mean, yerr=sem, fmt="-o", color=COHORT_C, lw=2.6, ms=7, capsize=4, zorder=4, label="cohort mean ± SEM")
        if example is not None:
            ax.plot(x, example[key], "-o", color=EX_C, lw=2.2, ms=6, zorder=5, label=f"{example_sub} (example)")
        ax.set_xticks(x); ax.set_xticklabels(EPOCHS); ax.set_xlim(-0.3, 2.3)
        ax.set_title(ttl, fontsize=11.5, weight="bold"); ax.set_ylabel(unit); ax.grid(axis="x", visible=False)
        if key == "rate":
            pc = wilcoxon_signed_rank_p_value(C[key]["chase"] - C[key]["pre"])[0]
            pp = wilcoxon_signed_rank_p_value(C[key]["post"] - C[key]["pre"])[0]
            ax.text(0.02, 0.02, f"exploratory p_unadj\npre→chase {pc:.3f}\npre→post {pp:.3f}", transform=ax.transAxes,
                    fontsize=8.5, va="bottom", color="#555")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"GoodCopBadCop bout statistics per epoch — n={n} recordings"
                 + (f" ({example_sub} highlighted)" if example is not None else ""),
                 fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02, "Bout rate = valid bouts / validly-tracked minute (denominator excludes tracking dropout). "
             "Robust bout-level metrics. Chase drop is chase-driven; pre-vs-post has a time-in-arena confound.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out, _ = save_standalone_exploratory_figure(
        fig,
        out,
        analysis_id="goodcopbadcop_bout_rate",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"wrote {out}  (cohort, n={n})")


# ------------------------------------------------------------------ single-fish figure
def plot_single(rid, zp, out):
    d = load(zp); rows = {ep: epoch_bouts(d, ep) for ep in EPOCHS}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3)); x = np.arange(3)
    # panel 1: bout rate bars
    ax = axes[0]
    rates = [rows[ep]["rate"] for ep in EPOCHS]
    ax.bar(x, rates, color=[COHORT_C, EX_C, COHORT_C], alpha=0.85)
    for xi, ep in zip(x, EPOCHS):
        ax.text(xi, (rows[ep]["rate"] or 0) + 1, f"{rows[ep]['n']} bouts\n{rows[ep]['valid_min']:.1f} min",
                ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_title(PANELS[0][1], fontsize=11.5, weight="bold"); ax.set_ylabel(PANELS[0][2])
    ax.set_xticks(x); ax.set_xticklabels(EPOCHS); ax.grid(axis="x", visible=False)
    # panels 2-4: distributions
    for ax, (key, ttl, unit) in zip(axes[1:], PANELS[1:]):
        data = [rows[ep][key] for ep in EPOCHS]
        data = [dd[np.isfinite(dd)] if dd.size else np.array([np.nan]) for dd in data]
        bp = ax.boxplot(data, positions=x, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.6))
        for patch, col in zip(bp["boxes"], [COHORT_C, EX_C, COHORT_C]):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        ax.set_title(ttl, fontsize=11.5, weight="bold"); ax.set_ylabel(unit)
        ax.set_xticks(x); ax.set_xticklabels(EPOCHS); ax.grid(axis="x", visible=False)
    fig.suptitle(f"Bout statistics per epoch — {rid.split('_GoodCop')[0]} (single fish)",
                 fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02, "Bout rate = valid bouts / validly-tracked minute. Boxes = per-bout distributions "
             "(median, IQR). Bouts from chaser_bout_response (robust, above the speed noise floor).",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out, _ = save_standalone_exploratory_figure(
        fig,
        out,
        analysis_id="goodcopbadcop_bout_rate",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"wrote {out}  (single: {rid})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording-id", default=None, help="single-fish mode for the matching recording")
    ap.add_argument("--example", default=DEFAULT_EXAMPLE, help="recording to highlight in cohort mode")
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    ap.add_argument("--tag", default="2026-07-17")
    args = parse_standalone_exploratory_args(
        ap,
        analysis_id="goodcopbadcop_bout_rate",
        argv=argv,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "font.family": "DejaVu Sans", "savefig.dpi": 160})
    recs = cohort()
    if not recs:
        raise SystemExit("No reachable GoodCopBadCop analysis zarrs.")
    if args.recording_id:
        hit = [(rid, zp) for rid, zp in recs if args.recording_id in rid]
        if not hit:
            raise SystemExit(f"No reachable recording matching {args.recording_id!r}.")
        rid, zp = hit[0]
        out = args.out_dir / f"goodcopbadcop_bout_rate_{rid.split('_GoodCop')[0]}_{args.tag}.png"
        plot_single(rid, zp, out)
    else:
        out = args.out_dir / f"goodcopbadcop_bout_rate_epochs_{args.tag}.png"
        plot_cohort(recs, args.example, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
