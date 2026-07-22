"""Trial-locked habituation figures: the escape response collapsing across chase trials.

    render_habituation_sheet(zarr_path)      -> a one-page PNG per fish
    render_habituation_cohort(registry, ...) -> the group figure

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
**The wall confound is drawn on the figure, never in a caption.**

The escape response collapses after chase trial 1-2. It is a real effect. But the fish also
moves to the wall after trial 1 and stays there, and at the wall it seldom escapes -- so a plot
of escape-rate-vs-trial, on its own, is indistinguishable from "the fish got cornered". Any
figure that shows the collapse without showing the wall lies by omission.

So the cohort sheet draws, on the same trial axis and directly beneath the collapse:

    * wall distance at trigger, falling 9.6 mm -> ~2 mm
    * the two controls that rule out the geometric trap:
        - with NO chaser at all, the wall does not suppress fast bouts (p=0.40)
        - on trial 1, fish that start AT the wall escape just as often (0.82 vs 0.75, p=0.68)

A reader who sees only the first panel would draw the wrong conclusion. That is a figure bug,
so it is fixed in the figure.

Other rules inherited from chaser_analysis_figures:
* Show the n -- every trial's fish count is annotated, and per-fish traces are drawn, not just
  a mean. 20/26 fish decline; 6 do not, and the reader should see them.
* Show the missing data -- dropout is a panel, because dropout rises when the fish freezes and
  freezing is the thing being measured.
* Rates are per *validly tracked* second. A trial that lost half its frames is not a trial
  where the fish escaped half as much.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path
from typing import Optional, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_io import load_chaser_distance_run  # noqa: E402
from fisheye.shared.arena_geometry import resolve_arena_geometry  # noqa: E402
from fisheye.visualization.chaser_analysis_figures import (  # noqa: E402
    _cohort_records,
    _latest_unsealed_inspection_child,
    _text,
)

warnings.filterwarnings("ignore")

ESCAPE_C = "#dc2626"     # escape -- red, the thing that collapses
FREEZE_C = "#2563eb"     # freeze -- blue, the thing that rises in its place
WALL_C = "#f59e0b"       # the confound -- amber, so it reads as a warning
FISH_C = "#cbd5e1"
CLEAN_DROPOUT = 0.05     # a "clean" trial: <5% of frames lost
WALL_BAND_MM = 5.0       # "at the wall"
EARLY = (1, 2)
LATE_MIN = 5


def _sem(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size > 1 else math.nan


class HabituationData:
    """One recording's trial-locked escape record, plus the two confound controls."""

    def __init__(self, zarr_path: Path) -> None:
        self.path = Path(zarr_path)
        root = zarr.open_group(str(self.path), mode="r", use_consolidated=False)
        self.root = root
        self.distance = load_chaser_distance_run(root)
        self.distance.require_derived_surface_authority("chaser_escape_events")
        self.recording_id = self.distance.recording_id
        self.run = root[self.distance.run_path]

        events = _latest_unsealed_inspection_child(
            self.run.get("chaser_escape_events")
        )
        if events is None:
            raise ValueError("No chaser_escape_events component; run it first.")
        if "trials" not in events or int(
            np.asarray(events["trials/ordinal"][:]).size
        ) == 0:
            raise ValueError(
                "chaser_escape_events has no trials "
                "(recording has no chase_trial_id)."
            )
        self.events = events

        trials = events["trials"]
        self.ordinal = np.asarray(trials["ordinal"][:], dtype=np.int64)
        self.rate = np.asarray(
            trials["escape_rate_per_valid_s"][:], dtype=np.float64
        )
        self.any_escape = np.asarray(trials["any_escape"][:], dtype=bool)
        self.escape_count = np.asarray(
            trials["escape_count"][:], dtype=np.int64
        )
        self.wall_mm = np.asarray(
            trials["wall_distance_at_trigger_mm"][:], dtype=np.float64
        )
        self.dropout = np.asarray(
            trials["dropout_fraction"][:], dtype=np.float64
        )
        self.latency = np.asarray(
            trials["first_escape_latency_s"][:], dtype=np.float64
        )
        self.trial_start = np.asarray(
            trials["start_frame"][:], dtype=np.int64
        )
        self.trial_end = np.asarray(trials["end_frame"][:], dtype=np.int64)
        self.trigger = np.asarray(trials["trigger_frame"][:], dtype=np.int64)
        self.event_frames = np.asarray(
            events["events/start_frame"][:], dtype=np.int64
        )
        self.fps = float(self.distance.fps)
        self.clean = self.dropout < CLEAN_DROPOUT
        self.freeze = self._freeze_per_trial()

    # -- freeze comes from the sibling canary, which already scores it per trial ----------
    def _freeze_per_trial(self) -> np.ndarray:
        out = np.full(self.ordinal.size, np.nan)
        if self.run.get("chaser_escape_freeze") is not None:
            self.distance.require_derived_surface_authority(
                "chaser_escape_freeze"
            )
        ef = _latest_unsealed_inspection_child(
            self.run.get("chaser_escape_freeze")
        )
        if ef is None or "trial_metrics" not in ef:
            return out
        m = ef["trial_metrics"]
        if "freeze_low_speed_fraction" not in m or "trial_ordinal" not in m:
            return out
        ordn = np.asarray(m["trial_ordinal"][:], dtype=np.int64).reshape(-1)
        frz = np.asarray(m["freeze_low_speed_fraction"][:], dtype=np.float64).reshape(-1)
        lut = {int(o): float(f) for o, f in zip(ordn, frz)}
        for i, o in enumerate(self.ordinal):
            out[i] = lut.get(int(o), np.nan)
        return out

    # -- CONTROL A: with no chaser at all, does the wall suppress fast bouts? -------------
    def pre_fast_bout_rate_by_wall(self, threshold_mm_s: float = 100.0) -> tuple[float, float]:
        """(rate at the wall, rate off the wall) per minute, in the pre epoch.

        If the wall physically suppressed fast bouts, this pair would differ. It does not
        (cohort: 3.72 vs 3.16 /min, p=0.40), which is half of why the habituation is not a
        geometric trap.
        """

        if self.run.get("chaser_bout_response") is not None:
            self.distance.require_derived_surface_authority(
                "chaser_bout_response"
            )
        bc = _latest_unsealed_inspection_child(
            self.run.get("chaser_bout_response")
        )
        if bc is None:
            return math.nan, math.nan
        eps = _text(bc["epochs/label_bytes"][:])
        if "pre_event" not in eps:
            return math.nan, math.nan
        e = eps.index("pre_event")
        lo = int(np.asarray(bc["epochs/start_frame"][:])[e])
        hi = int(np.asarray(bc["epochs/end_frame"][:])[e])

        wall = self._wall_trace()
        seg = wall[lo:hi]
        ok = np.isfinite(seg)
        t_near = float(np.count_nonzero(ok & (seg < WALL_BAND_MM))) / self.fps
        t_far = float(np.count_nonzero(ok & (seg >= WALL_BAND_MM))) / self.fps
        if t_near < 20.0 or t_far < 20.0:
            return math.nan, math.nan

        start = np.asarray(bc["bouts/start_frame"][:], dtype=np.int64)
        peak = np.asarray(bc["bouts/peak_speed_mm_s"][:], dtype=np.float64)
        valid = np.asarray(bc["bouts/valid"][:], dtype=bool)
        ei = np.asarray(bc["bouts/epoch_index"][:], dtype=np.int64)
        m = valid & (ei == e) & (start >= lo) & (start < hi) & (peak > float(threshold_mm_s))
        bw = wall[start[m]]
        n_near = int(np.count_nonzero(np.isfinite(bw) & (bw < WALL_BAND_MM)))
        n_far = int(np.count_nonzero(np.isfinite(bw) & (bw >= WALL_BAND_MM)))
        return n_near / t_near * 60.0, n_far / t_far * 60.0

    def _wall_trace(self) -> np.ndarray:
        ppm = float(self.distance.pixels_per_mm_projector)
        geo, _notes = resolve_arena_geometry(self.root, self.run, pixels_per_mm=ppm)
        if geo.radius_px is None or geo.center_x_px is None:
            raise ValueError("No circular arena geometry; cannot compute wall distance.")
        fish = np.asarray(
            self.distance.fish_centroid_arena_xy,
            dtype=np.float64,
        )
        fv = np.asarray(self.distance.fish_valid, dtype=bool)
        r = np.hypot(fish[:, 0] - geo.center_x_px, fish[:, 1] - geo.center_y_px)
        wall = (geo.radius_px - r) / ppm
        wall[~fv] = np.nan
        return wall

    # -- CONTROL B: on trial 1, does starting at the wall predict escape? -----------------
    def trial1_wall_and_escape(self) -> tuple[float, float]:
        i = np.flatnonzero(self.ordinal == 1)
        if i.size == 0:
            return math.nan, math.nan
        k = int(i[0])
        return float(self.wall_mm[k]), float(self.any_escape[k])

    def early_late(self, key: str) -> tuple[float, float]:
        """Clean-trial mean of `key` over trials 1-2 and trials 5+."""

        v = {"rate": self.rate, "freeze": self.freeze, "wall": self.wall_mm}[key]
        e = self.clean & (self.ordinal <= EARLY[1])
        l = self.clean & (self.ordinal >= LATE_MIN)
        with np.errstate(invalid="ignore"):
            a = float(np.nanmean(v[e])) if e.any() else math.nan
            b = float(np.nanmean(v[l])) if l.any() else math.nan
        return a, b


# ==========================================================================================
# Per-recording sheet
# ==========================================================================================


def render_habituation_sheet(zarr_path: Path, *, dpi: int = 130) -> bytes:
    d = HabituationData(zarr_path)
    fig = plt.figure(figsize=(13.5, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    fig.suptitle(f"{d.recording_id}  —  escape events per chase trial", fontsize=11)

    # -- the raster: every trial, every escape, on a common time-from-trigger axis --------
    ax = fig.add_subplot(gs[0, :])
    for i, o in enumerate(d.ordinal):
        lo, hi, tg = int(d.trial_start[i]), int(d.trial_end[i]), int(d.trigger[i])
        t0 = (lo - tg) / d.fps
        t1 = (hi - tg) / d.fps
        ax.plot([t0, t1], [o, o], color="#e2e8f0", lw=6, solid_capstyle="butt", zorder=1)
        if d.dropout[i] > CLEAN_DROPOUT:
            ax.plot([t0, t1], [o, o], color="#fca5a5", lw=6, alpha=0.45,
                    solid_capstyle="butt", zorder=2)
        inside = d.event_frames[(d.event_frames >= lo) & (d.event_frames <= hi)]
        if inside.size:
            ax.scatter((inside - tg) / d.fps, np.full(inside.size, o),
                       marker="|", s=110, color=ESCAPE_C, lw=1.8, zorder=4)
    ax.axvline(0.0, color="#0f172a", lw=1.2, zorder=3)
    ax.annotate("chaser reaches\ntrigger radius", xy=(0.0, 0.02), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=7, color="#0f172a", va="bottom")
    ax.invert_yaxis()
    ax.set_ylabel("chase trial")
    ax.set_xlabel("time from proximity trigger (s)")
    ax.set_title("red ticks = escapes;  pink = trial lost >5% of frames to tracking dropout",
                 fontsize=8, loc="left", color="#475569")

    # -- escape rate and THE CONFOUND, on the same axis ----------------------------------
    #
    # Dirty trials are drawn HOLLOW. The rate is per validly-tracked second, so a trial that
    # lost half its frames divides by a small denominator and can spike -- which is correct,
    # and which would badly mislead anyone reading a single fish's curve as if every point
    # were equally trustworthy. The cohort figure filters these out; here they are shown and
    # marked, because for one fish there may be nothing left after filtering.
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(d.ordinal, d.rate, color=ESCAPE_C, lw=1.8, zorder=2)
    clean, dirty = d.clean, ~d.clean
    ax.scatter(d.ordinal[clean], d.rate[clean], s=34, color=ESCAPE_C, zorder=3)
    ax.scatter(d.ordinal[dirty], d.rate[dirty], s=40, facecolor="white", edgecolor=ESCAPE_C,
               lw=1.4, zorder=3, label=f"dropout >{CLEAN_DROPOUT:.0%}")
    if dirty.any():
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    ax.set_xlabel("chase trial")
    ax.set_ylabel("escapes / validly-tracked second", color=ESCAPE_C)
    ax.tick_params(axis="y", labelcolor=ESCAPE_C)
    ax2 = ax.twinx()
    ax2.plot(d.ordinal, d.wall_mm, marker="s", ms=4, color=WALL_C, lw=1.4, ls="--", zorder=1)
    ax2.axhline(WALL_BAND_MM, color=WALL_C, lw=0.8, ls=":", alpha=0.7)
    ax2.set_ylabel("fish distance from wall at trigger (mm)", color=WALL_C)
    ax2.tick_params(axis="y", labelcolor=WALL_C)
    ax.set_title("the collapse — and the confound that rides along with it\n"
                 "hollow = dropout-heavy trial, rate is on a small denominator",
                 fontsize=8, loc="left")

    # -- freeze, the thing that replaces it ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if np.any(np.isfinite(d.freeze)):
        ax.plot(d.ordinal, d.freeze, marker="o", color=FREEZE_C, lw=1.8)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "no chaser_escape_freeze component\n(freeze not scored)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#64748b")
    ax.set_xlabel("chase trial")
    ax.set_ylabel("freeze fraction")
    e_r, l_r = d.early_late("rate")
    e_f, l_f = d.early_late("freeze")
    e_w, l_w = d.early_late("wall")
    ax.set_title(
        f"freeze replaces escape\n"
        f"trials 1-2 → 5+ :  escape {e_r:.2f}→{l_r:.2f} /s   "
        f"freeze {e_f:.2f}→{l_f:.2f}   wall {e_w:.1f}→{l_w:.1f} mm",
        fontsize=8, loc="left")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# ==========================================================================================
# Cohort
# ==========================================================================================


def _by_trial(data: Sequence[HabituationData], key: str, max_trial: int = 12,
              clean_only: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(trial axis, mean, sem, n fish) for one per-trial quantity."""

    xs = np.arange(1, max_trial + 1)
    mean = np.full(xs.size, np.nan)
    sem = np.full(xs.size, np.nan)
    n = np.zeros(xs.size, dtype=int)
    for j, o in enumerate(xs):
        vals = []
        for d in data:
            v = {"rate": d.rate, "freeze": d.freeze, "wall": d.wall_mm,
                 "any": d.any_escape.astype(float), "dropout": d.dropout}[key]
            m = d.ordinal == o
            if clean_only:
                m = m & d.clean
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            if vv.size:
                vals.append(float(np.mean(vv)))
        if vals:
            mean[j] = float(np.mean(vals))
            sem[j] = _sem(np.asarray(vals))
            n[j] = len(vals)
    return xs, mean, sem, n


def render_habituation_cohort(
    registry: Path,
    pattern: str = "%GoodCopBadCop%",
    *,
    dpi: int = 130,
) -> tuple[bytes, list[HabituationData], list[tuple[str, str]]]:
    """Returns (png, included, skipped). Skips are returned, never swallowed."""

    paths, skipped = _cohort_records(registry, pattern)
    data: list[HabituationData] = []
    for p in paths:
        try:
            data.append(HabituationData(p))
        except Exception as exc:
            skipped.append((p.name, f"{type(exc).__name__}: {exc}"))
    if not data:
        raise ValueError("No recordings with a trial-locked chaser_escape_events component.")

    from scipy import stats

    MAXT = 12                    # every trial panel shares one x-axis, or they cannot be read together
    fig = plt.figure(figsize=(16.5, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # ---- 1. THE COLLAPSE -----------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for d in data:
        m = d.clean & np.isfinite(d.rate) & (d.ordinal <= MAXT)
        ax.plot(d.ordinal[m], d.rate[m], color=FISH_C, lw=0.7, alpha=0.55, zorder=1)
    xs, mu, se, n = _by_trial(data, "rate", max_trial=MAXT)
    ax.errorbar(xs, mu, yerr=se, color=ESCAPE_C, lw=2.4, marker="o", capsize=3, zorder=3)
    ax.set_xlim(0.4, MAXT + 0.6)
    ax.set_xticks(range(2, MAXT + 1, 2))
    ax.set_xlabel("chase trial")
    ax.set_ylabel("escapes / validly-tracked second")
    # the n has to be visible without colliding with the axis: put it in axis-fraction space
    trans = ax.get_xaxis_transform()
    for x, m_, nn in zip(xs, mu, n):
        if np.isfinite(m_):
            ax.text(x, -0.11, str(nn), transform=trans, ha="center", va="top",
                    fontsize=6, color="#64748b", clip_on=False)
    ax.text(0.4, -0.11, "n =", transform=trans, ha="right", va="top", fontsize=6, color="#64748b")
    ax.set_title(f"1. the escape response COLLAPSES\n"
                 f"clean trials (<{CLEAN_DROPOUT:.0%} dropout); grey = each fish",
                 fontsize=9, loc="left")

    # ---- 2. FREEZE RISES IN ITS PLACE ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    xs, mu, se, _n = _by_trial(data, "freeze", max_trial=MAXT)
    ax.errorbar(xs, mu, yerr=se, color=FREEZE_C, lw=2.4, marker="o", capsize=3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.4, MAXT + 0.6)
    ax.set_xticks(range(2, MAXT + 1, 2))
    ax.set_xlabel("chase trial")
    ax.set_ylabel("freeze fraction")
    ax.set_title("2. …and freezing replaces it\nactive defence → passive defence", fontsize=9, loc="left")

    # ---- 3. THE CONFOUND. This panel is mandatory. ---------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    xs, mu, se, _n = _by_trial(data, "wall", max_trial=MAXT)
    ax.errorbar(xs, mu, yerr=se, color=WALL_C, lw=2.4, marker="s", capsize=3)
    ax.axhspan(0, WALL_BAND_MM, color=WALL_C, alpha=0.12)
    ax.text(MAXT * 0.55, WALL_BAND_MM * 0.4, "“at the wall”", fontsize=7, color="#92400e")
    ax.set_xlim(0.4, MAXT + 0.6)
    ax.set_xticks(range(2, MAXT + 1, 2))
    ax.set_xlabel("chase trial")
    ax.set_ylabel("fish distance from wall at trigger (mm)")
    ax.set_title("3. THE CONFOUND — the fish moves to the wall too,\n"
                 "and at the wall it seldom escapes",
                 fontsize=9, loc="left", color="#92400e")

    # ---- 4. CONTROL A: no chaser at all -> does the wall suppress fast bouts? ------------
    ax = fig.add_subplot(gs[1, 0])
    near, far = [], []
    for d in data:
        try:
            a, b = d.pre_fast_bout_rate_by_wall()
        except Exception:
            continue
        if np.isfinite(a) and np.isfinite(b):
            near.append(a)
            far.append(b)
    near = np.asarray(near)
    far = np.asarray(far)
    if near.size >= 3:
        for a, b in zip(near, far):
            ax.plot([0, 1], [a, b], color=FISH_C, lw=0.8, alpha=0.8, zorder=1)
        ax.plot([0, 1], [near.mean(), far.mean()], color="#0f172a", lw=2.6, marker="o", zorder=3)
        _t, p = stats.ttest_rel(near, far)
        ax.set_title(f"4. CONTROL — with NO chaser, does the wall suppress fast bouts?\n"
                     f"NO:  {near.mean():.2f} vs {far.mean():.2f} /min,  p={p:.2f}  (n={near.size})",
                     fontsize=9, loc="left", color="#166534")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["at the wall\n(<5 mm)", "off the wall\n(≥5 mm)"])
    ax.set_ylabel("fast-bout rate in the PRE epoch (/min)")
    ax.set_xlim(-0.3, 1.3)

    # ---- 5. CONTROL B: on trial 1, does starting at the wall predict escape? -------------
    ax = fig.add_subplot(gs[1, 1])
    w1, e1 = [], []
    for d in data:
        w, e = d.trial1_wall_and_escape()
        if np.isfinite(w):
            w1.append(w)
            e1.append(e)
    w1 = np.asarray(w1)
    e1 = np.asarray(e1)
    atw = w1 < WALL_BAND_MM
    if atw.sum() >= 3 and (~atw).sum() >= 3:
        pa, pb = float(e1[atw].mean()), float(e1[~atw].mean())
        _t, p = stats.ttest_ind(e1[atw], e1[~atw])
        ax.bar([0, 1], [pa, pb], color=[WALL_C, "#94a3b8"], alpha=0.9)
        for x, v, nn in ((0, pa, int(atw.sum())), (1, pb, int((~atw).sum()))):
            ax.text(x, v + 0.03, f"{v:.2f}\nn={nn}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title("5. CONTROL — on TRIAL 1, do fish already at the wall escape less?\n"
                     f"NO:  {pa:.2f} vs {pb:.2f},  p={p:.2f}. A fish at the wall CAN flee.",
                     fontsize=9, loc="left", color="#166534")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["started AT the wall", "started off the wall"])
    ax.set_ylabel("P(escape on trial 1)")

    # ---- 6. the paired effect, per fish --------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    A, B = [], []
    for d in data:
        a, b = d.early_late("rate")
        if np.isfinite(a) and np.isfinite(b):
            A.append(a)
            B.append(b)
    A = np.asarray(A)
    B = np.asarray(B)
    if A.size >= 3:
        for a, b in zip(A, B):
            ax.plot([0, 1], [a, b], color=ESCAPE_C if b < a else "#94a3b8",
                    lw=0.9, alpha=0.75, marker="o", ms=3)
        ax.plot([0, 1], [A.mean(), B.mean()], color="#0f172a", lw=2.8, marker="o", zorder=5)
        _t, p = stats.ttest_rel(A, B)
        w = stats.wilcoxon(A, B)
        ax.set_title(f"6. per fish: {A.mean():.2f} → {B.mean():.2f} escapes/valid-s\n"
                     f"p={p:.4f}  (Wilcoxon {w.pvalue:.4f});  {int((B < A).sum())}/{A.size} decline",
                     fontsize=9, loc="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["trials 1–2", "trials 5+"])
    ax.set_ylabel("escapes / validly-tracked second")
    ax.set_xlim(-0.3, 1.3)

    fig.suptitle(
        f"Trial-locked habituation of the escape response  —  {len(data)} fish, "
        f"{sum(d.ordinal.size for d in data)} chase trials\n"
        "The fish flees on the first one or two chases, then stops and freezes at the wall.   "
        "Panel 3 is the confound; panels 4–5 are why it is not a geometric trap.",
        fontsize=11,
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue(), data, skipped


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--recording-like", default="%GoodCopBadCop%")
    p.add_argument("--zarr", type=Path, default=None, help="Render a single recording sheet.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-recording", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.zarr:
        d = HabituationData(args.zarr)
        out = args.out_dir / f"{d.recording_id}_habituation.png"
        out.write_bytes(render_habituation_sheet(args.zarr))
        print(f"wrote {out}")
        return 0

    from fisheye.registry.db import RegistryPaths

    registry = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    png, data, skipped = render_habituation_cohort(registry, str(args.recording_like))
    out = args.out_dir / "habituation_cohort.png"
    out.write_bytes(png)
    print(f"wrote {out}   ({len(data)} recordings, {sum(d.ordinal.size for d in data)} trials)")
    for name, reason in skipped:
        print(f"  SKIPPED {name}: {reason}")

    if args.per_recording:
        for d in data:
            try:
                p = args.out_dir / f"{d.recording_id}_habituation.png"
                p.write_bytes(render_habituation_sheet(d.path))
                print(f"  wrote {p.name}")
            except Exception as exc:
                print(f"  FAILED {d.recording_id}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
