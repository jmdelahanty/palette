#!/usr/bin/env python
"""Reproducible: GoodCopBadCop habituation over chase trials (the within-training signal).

Per chase-trial ordinal: escape rate, freeze fraction, and first-escape latency, with a
cohort mean +/- SEM curve, faint per-recording lines, one highlighted example, and an
early-vs-late paired test. Escapes are bout peak > 100 mm/s (robust). Escape rate is per
validly-tracked second (dropout excluded). This is the best *learning* candidate.

Run (palette env):
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_habituation --exploratory-only
    scripts/py -m fisheye.analysis.analyze_goodcopbadcop_habituation --exploratory-only --example 21-50-10Z_arena_3

Key result (full canonical cohort): the within-training learning signal is STRONGLY
significant -- freeze fraction early ~0.41 -> late ~0.60 (p<0.001, n=29); escape rate
early ~0.62 -> late ~0.15 (p<0.001, n=28). This was only "plausible but underpowered"
on the n=11 June-14 slice (freeze p~0.005, escape p~0.21). On the full cohort it even
SURVIVES partialling out wall-distance (partial r=+0.23, p=0.003; see
analyze_goodcopbadcop_wall_mediator) -- wall-proximity is a chase mediator, and the
habituation is robust to it. This is the clearest learning result in the dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.analysis.goodcopbadcop_common import (
    figures_dir,
    parse_standalone_exploratory_args,
    resolve_cohort,
    save_standalone_exploratory_figure,
)
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

MAX_ORDINAL = 8
LATE_FROM = 5
DEFAULT_EXAMPLE = "21-50-10Z_arena_3"
COHORT_C = "#1b3a6b"
EX_C = "#c1435b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "font.family": "DejaVu Sans",
                     "savefig.dpi": 160})


def load(zp: str):
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_escape_events")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example", default=DEFAULT_EXAMPLE, help="Substring of the recording to highlight.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default="2026-07-17")
    args = parse_standalone_exploratory_args(
        ap,
        analysis_id="goodcopbadcop_habituation",
    )
    out_dir = args.out_dir or figures_dir()

    per_rec = []
    for rid, zp in resolve_cohort():
        try:
            esc, frz, latd = load(zp)
        except ChaserDistanceReadError:
            raise
        except Exception as ex:  # pragma: no cover
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        per_rec.append((rid, {"esc": esc, "frz": frz, "lat": latd}))

    ords = np.arange(1, MAX_ORDINAL + 1)

    def early_late(key):
        a, b = [], []
        for _, src in per_rec:
            s = src[key]
            ev = [s[o] for o in (1, 2) if o in s and np.isfinite(s[o])]
            lv = [s[o] for o in s if o >= LATE_FROM and np.isfinite(s[o])]
            if ev and lv:
                a.append(np.mean(ev))
                b.append(np.mean(lv))
        a, b = np.array(a), np.array(b)
        return np.mean(a), np.mean(b), wilcoxon_signed_rank_p_value(b - a)[0], len(a)

    panels = [("esc", "Escape rate", "escapes / valid s"),
              ("frz", "Freeze fraction", "low-speed fraction"),
              ("lat", "First-escape latency", "s")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (key, ttl, unit) in zip(axes, panels):
        agg = {o: [] for o in ords}
        for _, src in per_rec:
            s = src[key]
            xs = [o for o in ords if o in s and np.isfinite(s[o])]
            ax.plot(xs, [s[o] for o in xs], "-", color="#cccccc", lw=1, alpha=0.5, zorder=1)
            for o in xs:
                agg[o].append(s[o])
        mean = [np.mean(agg[o]) if len(agg[o]) >= 4 else np.nan for o in ords]
        sem = [np.std(agg[o]) / np.sqrt(len(agg[o])) if len(agg[o]) >= 4 else np.nan for o in ords]
        ax.errorbar(ords, mean, yerr=sem, fmt="-o", color=COHORT_C, lw=2.6, ms=6, capsize=3,
                    zorder=4, label="cohort mean ± SEM")
        ex = next((src[key] for rid, src in per_rec if args.example in rid), None)
        if ex:
            xs = [o for o in ords if o in ex and np.isfinite(ex[o])]
            ax.plot(xs, [ex[o] for o in xs], "-o", color=EX_C, lw=2, ms=5, zorder=5, label=args.example)
        e0, e1, p, n = early_late(key)
        ax.set_title(f"{ttl}\nearly {e0:.2f} → late {e1:.2f}  p={p:.3f} (n={n})", fontsize=11, weight="bold")
        ax.set_xlabel("chase trial (ordinal)")
        ax.set_ylabel(unit)
        ax.set_xticks(ords)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Habituation over chase trials -- the within-training learning signal",
                 fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02, "Escapes = bout peak > 100 mm/s. Rate per validly-tracked second (excludes dropout). "
             "Early = trials 1-2, late = trials 5+. Wall-distance is a mediator, not regressed here.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = out_dir / f"goodcopbadcop_habituation_{args.tag}.png"
    out, _ = save_standalone_exploratory_figure(
        fig,
        out,
        analysis_id="goodcopbadcop_habituation",
        bbox_inches="tight",
    )
    print("wrote", out, f" (n_rec={len(per_rec)})")
    for key in ("esc", "frz", "lat"):
        e0, e1, p, n = early_late(key)
        print(f"  {key}: early {e0:.3f} -> late {e1:.3f}  p={p:.3f}  n={n}")


if __name__ == "__main__":
    main()
