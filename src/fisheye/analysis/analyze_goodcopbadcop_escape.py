#!/usr/bin/env python
"""Reproducible: GoodCopBadCop acute escape response during the chase (the core finding).

Escape rate = number of fast bouts (per-bout peak speed > 100 mm/s) per validly-tracked
minute, computed pre vs chase. The per-bout peak is taken from `speed_smoothed_mm`
(track_kinematics) so it is above the raw-centroid noise floor; the bout-table
`peak_speed_mm_s` is reported as a cross-check.

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_escape

Key result (full canonical cohort, n=32): escape rate pre ~0.35 -> chase ~7.8 per
validly-tracked minute (~22x, 32/32, p<0.0001 on the smoothed peak; ~10x on the
bout-table peak). Strengthened from the earlier n=12 June-14 slice (~12x, p~0.0005).
This is the strongest, most robust GoodCopBadCop result: a proximity-triggered flee.
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr

from fisheye.analysis.goodcopbadcop_common import (
    resolve_cohort,
)
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

ESCAPE_PEAK_MM_S = 100.0


def load(zp: str) -> dict:
    r = zarr.open_group(zp, mode="r")
    distance = load_chaser_distance_run(r)
    distance.require_derived_surface_authority("chaser_bout_response")


def escape_rate(d: dict, epoch: str, peaks: np.ndarray) -> float:
    if epoch not in d["epochs"]:
        return np.nan
    s, e = d["epochs"][epoch]
    in_epoch = (d["onsets"] >= s) & (d["onsets"] <= e)
    n_escapes = int(np.count_nonzero(peaks[in_epoch] > ESCAPE_PEAK_MM_S))
    valid_minutes = d["moving_valid"][s:e + 1].sum() / d["fps"] / 60.0
    return n_escapes / valid_minutes if valid_minutes > 0.1 else np.nan


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cols = {"sm_pre": [], "sm_chase": [], "tab_pre": [], "tab_chase": []}
    for rid, zp in resolve_cohort():
        try:
            d = load(zp)
        except ChaserDistanceReadError:
            raise
        except Exception as ex:  # pragma: no cover - per-recording robustness
            print("skip", rid.split("_GoodCop")[0], ex)
            continue
        cols["sm_pre"].append(escape_rate(d, "pre", d["peak_smoothed"]))
        cols["sm_chase"].append(escape_rate(d, "chase", d["peak_smoothed"]))
        cols["tab_pre"].append(escape_rate(d, "pre", d["peak_table"]))
        cols["tab_chase"].append(escape_rate(d, "chase", d["peak_table"]))
    cols = {k: np.array(v, float) for k, v in cols.items()}

    def report(name: str, pre: np.ndarray, chase: np.ndarray) -> None:
        m = np.isfinite(pre) & np.isfinite(chase)
        pr, ch = pre[m], chase[m]
        ratio = np.mean(ch) / np.mean(pr) if np.mean(pr) > 0 else np.inf
        p = wilcoxon_signed_rank_p_value(ch - pr)[0]
        print(f"{name:34s} pre={np.mean(pr):.2f} chase={np.mean(ch):.2f} /valid-min  "
              f"{ratio:.1f}x  up={int((ch > pr).sum())}/{m.sum()}  p={p:.4f}")

    print(f"n={int(np.isfinite(cols['sm_chase']).sum())}  "
          f"escape = bout peak > {ESCAPE_PEAK_MM_S:g} mm/s, per validly-tracked minute\n")
    report("Escape rate [SMOOTHED peak]", cols["sm_pre"], cols["sm_chase"])
    report("Escape rate [bout-table peak]", cols["tab_pre"], cols["tab_chase"])


if __name__ == "__main__":
    main()
