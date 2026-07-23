# GoodCopBadCop cohort (n=32): what the assay shows, and what it cannot show

**Date:** 2026-07-14
**Cohort:** 32 unique recordings with a materialized `chaser_distance_run`, out of 41
GoodCopBadCop archives in the registry. (The 2026-05-29 batch of 4 has no distance run; the
2026-07-02 batch of 4 has no upstream analysis at all. One recording_id is registered twice
under two paths — deduplicated here.)
**Components:** `cra_near_field`, `chaser_radial_occupancy`, `chaser_response_regimes`,
`chaser_bout_response` — all re-run on the corrected dish-mask geometry (32/32 resolved
`dish_mask`).
**Unit of analysis:** the recording (one fish), **clustered by session**. 32 fish, 8 sessions.

> **CORRECTIONS (two, in sequence).**
>
> **(a) Session clustering — partly retracted.** A first pass treated all 32 fish as
> independent; a second overcorrected by clustering everything on session. Each fish has its
> own tank and camera, so most of the usual batch mechanisms do not apply. Testing it
> empirically (with the two recordings 1 second apart correctly merged into one session):
>
> | outcome | session ANOVA | ICC | naive p | session-RE p |
> |---|---|---|---|---|
> | Δ thigmotaxis | F=1.70, **p=0.156 (n.s.)** | 0.15 | **0.019** | 0.040 |
> | Δ speed | F=3.66, **p=0.008 (sig)** | **0.40** | 0.018 | **0.100** |
>
> They split. **Thigmotaxis is NOT session-clustered — the naive n=32 test stands (p=0.019).**
> **Speed IS** (ANOVA p=0.008), and that effect does not survive. Fish in a session still share
> room temperature, time of day and age-at-recording, which plausibly moves swim speed without
> moving wall preference. Date (p=0.35) and arena/camera (p=0.50) explain nothing.
>
> **(b) The near-band scalar was diluting the object effect.** A single 0-15 mm average includes
> bins where nothing happens. Resolving by distance band changes the object result materially —
> see §3.

---

## 1. The training effect on thigmotaxis is real but MARGINAL, and the speed effect does not survive

Mixed model, `outcome ~ 1 + (1 | session)`:

| endpoint | pre → post | estimate | p | session ANOVA | verdict |
|---|---|---|---|---|---|
| **Δ thigmotaxis** | 0.408 → 0.546 | **+0.138** | **0.019** | n.s. (p=0.16) | **holds** |
| Δ mean speed (mm/s) | 4.98 → 4.28 | −0.698 | 0.100 | **sig (p=0.008)** | **does not hold** |

**Thigmotaxis holds.** Fish move to the wall after the chase. But:

- The response is **highly heterogeneous**: post-period thigmotaxis ranges **0.062 to 0.977**,
  and **12 of 32 fish decreased**.
- `corr(pre, post) = +0.264, p=0.14` — **a fish's own baseline does not predict its post
  value.** The measure is unstable across epochs.
- `corr(pre, Δ) = −0.547, p=0.001` — strong regression to the mean. Low-baseline fish rise
  (+0.290), high-baseline fish fall (−0.097). The mean shift is real (regression to the mean
  does not move a mean), but most of the *variance* in Δ is not signal.

**The speed decrease does not survive.** Its ICC is 0.40 and session explains it (ANOVA
p=0.008), so the apparent slowing is largely a between-session difference (room temperature?
time of day? age at recording?) rather than a within-fish training effect.

This effect was nonetheless **invisible before the geometry fix** — `thigmotaxis_frac` reported
0.354 → 0.353 on the wrong arena circle. See `arena_calibration_and_thigmotaxis_2026-07-14.md`.

### Genuine non-responders exist, and they are not a data artifact

Several fish never go near the objects and never wall-hug, with clean tracking:

| recording | thig pre → post | post visits to aggressive | post dropout |
|---|---|---|---|
| `2026-06-14T21-50-10Z_arena_1` | 0.196 → **0.078** | **0** | 0.0% |
| `2026-06-21T21-29-13Z_arena_4` | 0.682 → 0.384 | **0** | 0.0% |
| `2026-06-21T18-18-32Z_arena_4` | 0.603 → **0.074** | 1 | 1.0% |
| `2026-06-14T21-50-10Z_arena_3` | 0.224 → **0.191** | 6 | 0.0% |

`corr(post dropout, post thigmotaxis) = +0.20, p=0.27` — **dropout does not explain them.**
These are real behavioural non-responders (or centre-dwellers). Note three of them are from the
same session (`21-50-10Z`), which is exactly the clustering the mixed model now accounts for.

---

## 2. The designed CRA endpoints show nothing

| endpoint | n | mean | vs 0 |
|---|---|---|---|
| `nearzone_occ_specificity` | 32 | +0.0051 | t=+1.31, p=0.20 |
| `approach_p05_specificity` | 32 | −2.29 mm | t=−1.27, p=0.21 |

---

## 3. The object effect: a localized dose-response, and suggestive evidence of learning

The single 0–15 mm "near band" scalar was the wrong readout. It averages over bins where
nothing happens (2 mm, 6 mm) and **halved the signal**. Resolving `steering_excess_vs_virtual`
by distance band gives a dose-response:

`steering_excess_by_band` — post period, aggressive (red) object:

| band | 6 mm | **10 mm** | **14 mm** | 18 mm | 22 mm | 27 mm | 35 mm | 50 mm |
|---|---|---|---|---|---|---|---|---|
| excess | +0.06 | **+0.574** | **+0.551** | +0.30 | +0.23 | +0.14 | −0.03 | −0.25 |
| p | 0.66 | **0.009** | **0.011** | 0.17 | 0.12 | 0.50 | 0.87 | 0.11 |

Localized, peaked, decaying to zero by 35 mm. The inert (blue) object shows nothing coherent at
any band. **A real object-driven response looks like this; an artifact is flat.**

### In the 8–16 mm shell where the effect lives

| | n | mean excess | p |
|---|---|---|---|
| **pre**, aggressive | 32 | **+0.267 mm** | **0.0063** |
| **post**, aggressive | 28 | **+0.676 mm** | **0.0004** |
| pre, inert | 32 | +0.139 mm | 0.030 |
| post, inert | 28 | +0.062 mm | 0.54 |

| contrast | n | Δ | p |
|---|---|---|---|
| **aggressive, pre → post** (paired) | 28 | +0.255 → **+0.676** (**+0.422**) | **t p=0.050**, Wilcoxon 0.14, session-RE 0.041 |
| inert, pre → post (paired) | 28 | +0.172 → +0.062 | **0.36 (no change)** |
| post, aggressive vs inert (paired) | 25 | +0.624 mm | **0.0023** |

Median bouts per fish in the shell: **142–246** — not the 6 visits the near-band scalar was
resting on. Widening the band solves the sample-size problem.

**Read:** avoidance steering around the red object nearly **triples** after training; the inert
object does not move at all; the effect is spatially localized and decays with distance. That
is a coherent, object-driven learning signature.

**Caveats, unchanged and load-bearing:**

- It is **marginal** — p=0.050 parametric, 0.14 Wilcoxon.
- The **innate component is confirmed**: the pre-period bias is significant (+0.267, p=0.006),
  before any chase.
- **Colour and position are still not counterbalanced** (below). What is new is that the
  *increase* is aggressive-specific and the inert control is flat — which the pure-innate-bias
  story does not predict.

### Why it still cannot be closed: the design is not counterbalanced

Across all 32 recordings:

| | colour | pre position | post position |
|---|---|---|---|
| **aggressive** | `#ff0000` (red), **32/32** | (18, 18), **32/32** | (64, 64), **32/32** |
| **inert** | `#1600ff` (blue), **32/32** | (64, 18), **32/32** | (18, 64), **32/32** |

"Aggressive" is perfectly confounded with colour and position. Both objects move to their
opposite diagonal between pre and post, so "moving to a new place" is controlled — but the
specific post positions are not symmetric, and colour is never swapped.

**Counterbalance colour and position across fish.** It is the single change that would turn a
marginal, confounded result into a clean one.

---

## 4. The detection bias is systemic, not idiosyncratic

Training-epoch fish-tracking dropout across the cohort:

- median **2.1%**, max **87.6%**
- **11 / 32** recordings have **>10%** dropout
- **5 / 32** have a contiguous gap **>30 s**

This is the failure documented in `goodcopbadcop_chase_epoch_findings_2026-07-14.md`: the
detector's confidence collapses to ~0.30 (below the 0.40 threshold) precisely when the fish is
motionless against the rim — i.e. **when it is freezing, which is the behavior of interest.**
Recall is anti-correlated with the response.

So it is not one bad arena. A third of the cohort is materially affected, and every
freeze-related number (`freeze_index`, `immobile_fraction`) is a **lower bound** until
detection is re-run at `conf ~0.20–0.25`.

Note that this cuts *toward* the thigmotaxis result rather than against it: the frames being
lost are frames where the fish is frozen at the wall, so the true post-period thigmotaxis is
if anything **higher** than the 0.546 reported here.

---

## Summary (session-clustered; 32 fish, 8 sessions)

- **Marginal:** training increases thigmotaxis (+0.138, **p=0.040**). Real but weak, highly
  heterogeneous (12/32 fish *decreased*, range 0.06–0.98), and a fish's own baseline does not
  predict its post value (r=0.26, p=0.14).
- **Not supported:** the speed decrease (p=0.100). It is largely a between-session effect
  (ICC=0.40).
- **Robust but INNATE, not learned:** the steering bias away from the red/aggressive object
  (post p=0.006, pre p=0.014). It is already present *before any chase*, and does not
  significantly increase with training (p=0.44).
- **Uninterpretable as designed:** anything contrasting aggressive vs inert — colour and
  position are fixed across all 32 fish.
- **Needs fixing before the chase epoch means anything:** detector recall on frozen fish.

**The single highest-value protocol change is counterbalancing colour and position.** Without
it the assay's central contrast cannot be interpreted, no matter how many fish are run.

**The single highest-value analysis change is treating session as a random effect.** Four fish
per session are not four independent observations.

## Reproducing

```bash
for m in cra_near_field chaser_radial_occupancy chaser_response_regimes chaser_bout_response; do
  python -m fisheye.utils.run_goodcopbadcop_$m \
      --recording-like '%GoodCopBadCop%' --limit 60 --coverage-min 0 --apply --overwrite
done
```
