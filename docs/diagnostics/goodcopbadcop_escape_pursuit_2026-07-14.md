# The escape response is real, large, and the chaser reels the fish back in

**Date:** 2026-07-14
**Cohort:** the same 32 GoodCopBadCop recordings with a materialized `chaser_distance_run`.
**Source:** `chaser_bout_response`'s bout table (`peak_speed_mm_s`, `start_frame`) plus the
distance run's per-frame `distances/distance_mm`. **No new upstream stages, no megabouts.**
**Unit of analysis:** the recording (one fish). Every number below is a per-fish test.

---

## The headline

During the 180 s chase, fish escape **11×** more often than at baseline, they escape **when the
chaser gets close**, the escape is **directed away from the chaser** and gains ~8 mm — and then
**the chaser takes most of it back within four seconds.**

All numbers below are re-derived from the materialized `chaser_escape_events` component
(`docs/chaser_escape_events_contract.md`), not from a scratch script.

| result | value | p |
|---|---|---|
| escape rate, pre → chase | 3.17 → **35.84** per *validly-tracked* min (**11.3×**) | **4.0e-05** |
| escape onset distance vs ordinary bouts | **−6.68 mm** closer | **0.0005** |
| distance gained by +0.5 s | **+8.38 mm** | **1.1e-10** |
| ground lost back, +0.5 s → +4 s | **+6.60 mm** (21/32 fish) | **0.011** |
| same, against the **static** dot | −2.25 mm (it keeps the ground) | 0.042 |
| **pursuit: chase loses more ground than the static control** | **+10.03 mm** | **0.0004** |
| *(net change, onset → +4 s — the statistic that lies)* | *+3.70 mm* | *0.15* |

31 of 32 fish escape more during the chase.

**The single cleanest control:** against the static dot, the same fish making the same fast bout
gains **+0.50 mm, p=0.46 — nothing.** A fast bout aimed at nothing goes nowhere. Only against
the chaser is it directed. That contrast is within-fish and within-object.

> **On the denominator.** The rate is per minute of *validly tracked* time, not wall clock.
> Detector recall collapses on frozen fish (11/32 recordings lose >10% of the chase epoch), so
> a wall-clock rate scores a fish that vanished for 80 s as having escaped *less*. On wall clock
> the same effect reads 3.10 → 24.07/min (7.8×); the valid-time correction raises it to 11.3×.
> This also moves post-vs-pre escape rate from p=0.063 to **p=0.038**. That one is a marginal
> result that flipped on a methodological choice, and **no claim of retained sensitization
> should rest on it.**

---

## 1. How this was nearly missed: two methodological failures worth recording

### (a) K-means cannot see a 1.5% behavioural class

The first attempt clustered 322,781 bouts (K=4, on turn/speed/displacement/duration/tortuosity)
and asked whether an "escape-like" cluster rose during the chase. It reported **+0.031, p=0.057
— a null.**

That was an artifact of the method. Escape bouts are **1.45%** of all bouts. K-means minimizes
within-cluster variance, so it will never spend a centroid on 1.5% of the mass; the escapes were
absorbed into a generic "high-angle turn" cluster (57°, 36 mm/s) whose *rate* barely moves. A
7.8× effect diluted to nothing.

**A one-line speed threshold found it immediately.** Unsupervised repertoire discovery is the
wrong instrument for a rare, high-amplitude class you can already name. Reach for it when you
*don't* know what you're looking for.

### (b) A net-change statistic that sums two opposite-signed processes has no power

The first per-fish pursuit test measured net Δdistance from escape onset to +4 s and returned
**+1.55 mm, p=0.55 — a null**, which was briefly reported as a failure to replicate the pooled
trace.

It is not a null. Escape (+8.4 mm) and recapture (−5.5 mm) have opposite signs and nearly cancel.
Decomposing into **gain (onset → +0.5 s)** and **recapture (+0.5 s → +4 s)** recovers both at
p=3.6e-10 and p=0.029.

Before accepting a null, check that the statistic could have detected the effect if it were there.

### (c) The pooled trace *was* pseudoreplicated — that part of the criticism stands

The first version of the pursuit figure pooled 2,097 escape **events** and took a per-timepoint
median. That weights high-escape fish heavily and treats events from one fish as independent.
The result survives per-fish weighting, but the pooled trace overstated it (it showed the fish
ending up *closer* than it started; per-fish, it ends up +2.95 mm away, n.s.).

---

## 2. The escape is triggered by proximity, and it is directed

Per-fish medians, chase epoch:

- ordinary bout onset: **20.1 mm** from the aggressive chaser
- escape bout onset: **13.5 mm** (paired **p=0.0005**)

At escape onset the chaser is actively closing (Δ = −0.48 mm vs the preceding second, p=0.008).
The trigger band (10–14 mm) is **the same band where the steering dose-response peaks**
(`steering_excess_by_band`, +0.574 at 10 mm, p=0.009). Two independent measures converge on the
same distance.

**The escape is directed, and the static-dot control proves it.** Per-fish, baseline-subtracted
at escape onset:

| t (s) | CHASE Δmm | p vs 0 | POST (static dot) Δmm | p vs 0 | chase vs post |
|---|---|---|---|---|---|
| 0.0 | −0.48 | 0.008 | 0.00 | 0.18 | **0.013** |
| +0.2 | **+6.38** | <1e-4 | +0.32 | 0.61 | **<1e-4** |
| +0.6 | **+8.15** | <1e-4 | +0.69 | 0.45 | **0.0001** |
| +1.0 | **+10.39** | <1e-4 | +1.46 | 0.21 | **0.0001** |
| +1.5 | +7.96 | <1e-4 | +2.34 | 0.19 | **0.031** |
| +2.0 | +6.44 | 0.004 | +2.63 | 0.16 | 0.10 |
| +3.0 | +4.60 | 0.056 | +3.44 | 0.097 | 0.51 |
| +4.0 | +2.95 | 0.25 | +3.76 | 0.10 | 0.91 |

An equally fast bout in the post period, made by the same fish against a static dot, produces
**no distance gain at all** (+0.32 mm at 0.2 s, p=0.61). It is a fast bout, not an escape — it
isn't aimed at anything. Only with the object as the reference frame is the difference visible.

## 3. The pursuit

Decomposed per fish:

| phase | chase | static-dot post |
|---|---|---|
| gain, onset → +0.5 s | **+8.41 mm** (p=3.6e-10) | +0.40 mm (n.s.) |
| ground lost back, +0.5 s → +4 s | **+5.45 mm** (p=0.029, 24/32 fish) | −3.28 mm (p=0.062, 10/29 fish) |

**Paired contrast: the chase loses 8.48 mm more ground than post, p=0.0097 (n=29).**

The fish flees, gains ~8 mm, and the chaser closes ~5.5 mm of it back inside four seconds; the
static dot cannot, so against it the gain persists. **That difference is the pursuit, isolated
within-fish.** It is the mechanism behind the operator's own description: *"they escape by trying
to run away but the chaser will follow them."*

---

## 4. Controls

### Threshold sweep — the effect does not depend on a lenient cutoff

| threshold | median escapes/fish (chase) | pre /min | chase /min | ratio | p | onset esc vs ord |
|---|---|---|---|---|---|---|
| 100 mm/s | 60 | 3.10 | 24.07 | **7.8×** | 1.9e-09 | 13.5 vs 20.1 (p=0.0005) |
| 150 mm/s | 49 | 1.17 | 18.73 | **16.0×** | 7.0e-10 | 12.7 vs 19.9 (p=0.0002) |
| 200 mm/s | **3** | 0.15 | 1.79 | **11.7×** | 3.0e-04 | 11.8 vs 19.8 (p=0.016) |
| 300 mm/s | 0 | 0.03 | 0.00 | — | n.s. | — |

At 200 mm/s the median fish shows **3** escapes in the chase, matching the operator's independent
recollection of "once or twice." The rate ratio and the proximity trigger hold across the whole
usable range.

**Caveat:** nothing exceeds ~300 mm/s. Larval C-starts peak at 100–500 mm/s, and a C-start lasts
~15–20 ms = **1.5–2 frames at 100 fps**. Centroid-differenced peak speed almost certainly
*under-reads* true escape velocity, and the absolute mm/s values here should not be compared to
the high-speed-imaging literature. The **contrasts** are unaffected.

### Tracking dropout does not explain it

A lost-and-regained fish produces a spurious fast bout, and dropout is worst exactly during the
chase. It is not the cause:

- `corr(training dropout, escape rate)` = **r=−0.11, p=0.55** (Spearman rho=+0.13, p=0.47)
- **Clean-tracking subset only** (training dropout <5%, n=21 fish): **2.89 → 22.18 /min, 7.7×,
  p=3.0e-07** — identical to the full cohort.

---

## 4b. The escape response habituates: active defence → passive defence, in ~10 s

The chase epoch is **not 180 s of continuous chasing** — it is ~12 experimenter-initiated trials
of ~5 s, delimited by the controller's `chase_trial_id`. **394 chase trials across 32 fish.**
Resolving by trial changes the picture completely; averaged over the epoch this is invisible.

Clean trials only (<5% dropout, 311/394 survive):

| trial | n | P(escape) | escapes/valid-s | freeze | wall dist at trigger |
|---|---|---|---|---|---|
| **1** | 25 | **0.72** | **3.95** | 0.33 | **9.6 mm** |
| **2** | 26 | 0.35 | 1.77 | 0.53 | 3.8 mm |
| 3 | 24 | 0.08 | 0.48 | 0.63 | 1.7 mm |
| 4–12 | ~24 ea | ~0.10 | ~0.5 | ~0.65 | ~2.0 mm |

**Paired within fish, trials 1–2 vs 5+ (n=26):**

| | trials 1–2 | trials 5+ | p |
|---|---|---|---|
| escapes / valid second | **2.66** | **0.57** | **0.0001** (20/26 decline) |
| freeze fraction | 0.45 | 0.65 | **4e-05** |

Within-fish slope of escape rate on trial number: **−0.184 /trial, p=0.011, 24/29 fish negative.**

Escape latency is **+0.24 s after the proximity trigger** (88% of escapes fire *after* it), so
the causal ordering is right: the chaser closes, then the fish flees.

This is precisely the operator's independent recollection — *"many of the fish will only escape
once or twice throughout the entire session"* — and it explains the 62% freeze during the chase
(§6): after trial 2, the fish has stopped running.

### The tracking-dropout artifact does NOT explain it

Dropout rises across trials in the marginal table (0.05 → 0.20), and escapes are detected from
bouts, which need tracking. Three checks:

- **Dropout has no within-fish trend**: +0.007/trial, **p=0.31** (19/32 negative). The marginal
  rise is a *between*-fish effect — a few bad fish weighting later trials.
- **Normalizing by validly-tracked time**: the decline survives (−0.18/trial, p=0.011).
- **Clean trials only**: everything survives (escape rate p=0.018, P(escape) p=0.045, freeze
  p=0.040).

### The "cornered fish" alternative — ruled out, but read this before you quote the result

The fish **moves to the wall after trial 1 and stays** (9.6 mm → ~2 mm), and at the wall it
seldom escapes: **P(escape) = 0.10 at the wall vs 0.60 off it (p<0.0001)**. So "habituation" and
"pinned against the wall with nowhere to flee" are **confounded by construction.**

Two checks rule out the geometric trap:

| check | result |
|---|---|
| With **no chaser at all** (pre epoch), does the wall suppress fast bouts? | **No.** 3.72/min at the wall vs 3.16/min off it, **p=0.40** |
| On **trial 1**, do fish that start *at* the wall escape less? | **No.** 0.82 (n=11) vs 0.71 (n=21), **p=0.53** |

A fish at the wall escapes perfectly well — when it still wants to. So a fish sitting at the wall
on trial 5 and not fleeing is not a fish that *cannot* flee. **Wall position, freezing, and escape
failure are three faces of one defensive state switch, not three findings.**

**Do not regress wall distance out.** It is *downstream of the chase* — a mediator of the
response, not a nuisance covariate — so controlling for it is conditioning on a post-treatment
variable. `trials/wall_distance_at_trigger_mm` is stored so the question can be asked, not
silently absorbed.

**The honest limit:** the direct test (does the decline survive when the fish is off the wall?) is
**underpowered to the point of uninformative** — only 47 off-wall clean trials cohort-wide, and
just **3 fish** have both early and late off-wall trials (p=0.84 on n=3). That is absence of
evidence, not evidence of absence. The two indirect checks above are what carry the argument.

---

## 5. What this does NOT show

- **Nothing about aggressive vs inert.** Only the aggressive object chases, so there is no inert
  comparison to be made for escape. The colour/position confound
  (`goodcopbadcop_cohort_results_2026-07-14.md` §3) is untouched by this analysis.
- **Post-period escape rate is only marginally elevated** (3.10 → 4.56 /min, **p=0.063**). There
  is a *hint* of retained sensitization to the now-static red dot, and it does not reach
  significance. Do not claim it.
- **No claim about escape kinematics.** At 100 fps we cannot resolve C-start latency, angle, or
  true peak velocity. This is an event-rate and trajectory result, not a kinematic one.

## 6. The coarse repertoire result, for the record

Separately from the escapes, the 4-cluster repertoire *does* shift near objects in the post
period versus wall-matched virtual controls — but **it is not specific to the aggressive object**:

| category (post, 8–16 mm shell) | aggressive excess | inert excess | agg vs inert |
|---|---|---|---|
| scoot | −0.065 (p=0.0045) | −0.059 (p=0.0042) | **p=0.62** |
| slow/small | +0.032 (p=0.014) | +0.036 (p=0.0045) | **p=0.83** |
| high-angle | +0.040 (p=0.039) | −0.005 (p=0.72) | p=0.10 |

After training the fish stops scooting and slows down near **an object** — the blue inert dot as
much as the red one. That is a generalized object response, not a learned threat response. Only
high-angle turns lean aggressive-specific, at p=0.10 with n=27. **A hint, not a result.**

The one genuinely informative repertoire finding is what happens *during* the chase: routine
scoots collapse (0.576 → 0.218) and are replaced 12-fold by near-zero-displacement "jitter" bouts
(0.028 → 0.335, both p<1e-4). That is a **freeze** signature, not an escape one — consistent with
the 62% immobile fraction at 2–4 mm. The fish's dominant strategy is to freeze; the escape is the
rare, large, proximity-triggered exception.

---

## Reproducing

The escape analysis is now a component. Contract:
`docs/chaser_escape_events_contract.md`.

```bash
# requires chaser_bout_response to be materialized first
python -m fisheye.utils.run_goodcopbadcop_chaser_escape_events \
    --recording-like '%GoodCopBadCop%' --limit 60 --coverage-min 0 --apply --overwrite
```

Then read, per recording, from
`analysis/chaser_distance_runs/<run>/chaser_escape_events/<component>/`:

- `rates/escape_rate_per_valid_min` — (epoch,)
- `trigger/proximity_shift_mm` — (epoch, reference)
- `pursuit/{gain_mm, recapture_mm}` — (epoch, reference). **Not `net_mm`.**
- `traces/delta_distance_mm` — (epoch, reference, time)

Map reference → aggressive/inert via `cra_primary_endpoint/<v>/objects/object_role_code`,
**never** index order. `pursuit/` and `traces/` are already per-recording medians and are the
safe things to aggregate; `events/` is one row per escape and must be reduced per fish first.

The coarse-repertoire analysis (§6) remains a scratch script under `scratchpad/repertoire/` — it
produced a negative and was not promoted to a component.
