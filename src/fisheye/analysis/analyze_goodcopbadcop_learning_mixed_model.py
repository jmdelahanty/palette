#!/usr/bin/env python
"""Reproducible: do the per-fish GoodCopBadCop learning metrics survive a session random effect?

The per-fish learned-avoidance metric occ_did (aggressive-vs-inert diff-in-diff) flagged a
handful of strong "learners", but 4 of the top 6 were one session (21-29-13) -- a batch
confound. This fits a random-intercept model for each metric:

    metric ~ 1 + (1 | session)

and reports (a) the population mean (fixed intercept) with a p-value that accounts for
session clustering, (b) the intraclass correlation ICC = session_var / (session_var +
residual_var) -- how much of the between-fish variance is really between-session batch, and
(c) per-session means, plus a leave-out-21-29-13 sensitivity check.

Run alongside the freeze-learning index (frz_delta) as a positive control: a genuinely
individual, near-universal learned signal should keep a nonzero population mean under the
session RE, whereas a batch-driven one collapses.

Run (palette env):
    python -m fisheye.analysis.analyze_goodcopbadcop_learning_mixed_model

Reads the canonical registry (goodcopbadcop_common).
"""
from __future__ import annotations

import argparse
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from fisheye.analysis.goodcopbadcop_common import resolve_cohort
from fisheye.analysis.analyze_goodcopbadcop_per_fish import spatial_avoidance_did, learning_index
from fisheye.group_statistics.paired import wilcoxon_signed_rank_p_value

BATCH_SESSION = "2026-06-21T21-29-13Z"
METRICS = [
    ("occ_did", "learned object avoidance (agg-vs-inert diff-in-diff)"),
    ("frz_delta", "freeze-learning index (late - early)  [positive control]"),
]


def session_of(rid):
    return re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)", rid).group(1)


def build_frame():
    recs = []
    for rid, zp in resolve_cohort():
        row = {"recording_id": rid, "session": session_of(rid)}
        try:
            row.update(spatial_avoidance_did(zp))
        except Exception:
            pass
        try:
            row.update(learning_index(zp))
        except Exception:
            pass
        recs.append(row)
    return pd.DataFrame(recs)


def fit_random_intercept(df, metric):
    d = df[["session", metric]].dropna().rename(columns={metric: "y"})
    n, n_sess = len(d), d["session"].nunique()
    y = d["y"].to_numpy()
    naive_p = wilcoxon_signed_rank_p_value(y)[0] if y.size >= 6 else np.nan
    out = {"n": n, "n_sess": n_sess, "naive_mean": float(np.mean(y)), "naive_p": naive_p, "collapsed": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm("y ~ 1", d, groups=d["session"]).fit(reml=True, method="lbfgs")
        group_var = float(res.cov_re.iloc[0, 0])
        resid = float(res.scale)
        out.update(intercept=float(res.fe_params["Intercept"]), p=float(res.pvalues["Intercept"]),
                   group_var=group_var, resid=resid,
                   icc=group_var / (group_var + resid) if (group_var + resid) > 0 else np.nan)
    except Exception:
        # RE variance collapsed to ~0 (no between-session structure): reduces to OLS/one-sample t.
        t = stats.ttest_1samp(y, 0.0)
        out.update(intercept=float(np.mean(y)), p=float(t.pvalue), group_var=0.0,
                   resid=float(np.var(y, ddof=1)), icc=0.0, collapsed=True)
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = build_frame()

    for metric, label in METRICS:
        print(f"\n=== {metric}: {label} ===")
        r = fit_random_intercept(df, metric)
        print(f"  n={r['n']} fish across {r['n_sess']} sessions")
        print(f"  naive (ignores session): mean={r['naive_mean']:+.4f}  Wilcoxon p={r['naive_p']:.3f}")
        print(f"  random-intercept model  metric ~ 1 + (1|session):")
        print(f"    population mean (fixed intercept) = {r['intercept']:+.4f}   p={r['p']:.3f}"
              f"   {'(!= 0)' if r['p'] < 0.05 else '(n.s. -- no population-level effect)'}")
        print(f"    session var={r['group_var']:.5f}  residual var={r['resid']:.5f}  "
              f"ICC={r['icc']:.2f}  ({r['icc']*100:.0f}% of variance is between-session batch)")
        # per-session means
        g = df.groupby("session")[metric].agg(["mean", "count"]).dropna()
        print("    per-session mean:")
        for sess, row in g.sort_values("mean", ascending=False).iterrows():
            flag = "  <-- the batch" if sess == BATCH_SESSION else ""
            print(f"      {sess}  mean={row['mean']:+.4f}  n={int(row['count'])}{flag}")
        # sensitivity: drop the batch session
        d2 = df[df["session"] != BATCH_SESSION]
        r2 = fit_random_intercept(d2, metric)
        note = " [RE var~0 -> pooled t]" if r2["collapsed"] else ""
        print(f"    leave-out {BATCH_SESSION}: mean={r2['intercept']:+.4f} p={r2['p']:.3f} "
              f"(n={r2['n']}, {r2['n_sess']} sessions){note}")


if __name__ == "__main__":
    main()
