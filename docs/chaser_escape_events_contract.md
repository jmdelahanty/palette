# `chaser_escape_events` contract

**Schema:** `palette.chaser_escape_events.v3`
**Path:** `analysis/chaser_distance_runs/<run>/chaser_escape_events/<component>`
**Module:** `src/fisheye/analysis/chaser_escape_events.py`
**Runner:** `python -m fisheye.utils.run_goodcopbadcop_chaser_escape_events`
**Tests:** `tests/unit/fisheye/test_chaser_escape_events.py`

## What it is for

An escape is a **rare, high-amplitude event**, not a shift in the bout distribution. This
component picks the fast bouts out of the `chaser_bout_response` bout table and asks four
things about them:

1. **Rate** — how often, per minute of *validly tracked* time, in each epoch?
2. **Trigger** — how far was the object when the escape fired, versus an ordinary bout?
3. **Pursuit** — aligned on escape onset, does the fish gain ground, and does the chaser
   take it back?
4. **Habituation** — per chase *trial*, does the escape response collapse?

## Hard dependency

Requires a materialized **`chaser_bout_response`** component on the same distance run. It
supplies the bout table *and* the virtual references, which are read back and reconstructed
from their stored rotation angles — not recomputed from this module's own defaults. That is
deliberate: recomputing would let the two components' virtual twins drift apart silently
while every number still looked plausible. Pinned by
`test_references_are_inherited_from_the_bout_component_not_recomputed`.

Missing parent → a `ValueError` naming `chaser_bout_response`. It does not fall back.

## Escape definition

A valid bout whose `peak_speed_mm_s` exceeds `peak_speed_threshold_mm_s` (default **100.0**).

**The escape classification is a pure velocity cut — nothing else.** No proximity, no C-start
shape. A fast *forward* swim and a turn-away both count if they clear the threshold; the 13.5 mm
trigger ring is *descriptive*, not classificatory.

### The turn tier (on by default)

Because velocity alone conflates two things, every escape is additionally tagged with a **turn
tier**: an escape is **`high_turn`** (a C-start-like reorientation) when its `|turn_deg|` also
reaches `high_turn_threshold_deg` (default **45°**); otherwise it is a straight forward
**`dash`**. This is reported by default (`events/is_high_turn`, `rates/high_turn_escape_count`,
`rates/high_turn_escape_rate_per_valid_min`, `rates/high_turn_fraction`) and **never changes
which bouts are escapes** — escape stays speed-only, so the headline rate/pursuit/habituation
numbers are untouched.

The tier is well-motivated: high turn is ~4× enriched in escapes over ordinary bouts (21% vs 5%
at ≥60°), and the high-turn sub-population is *if anything more chase-specific* than the velocity
cut alone (~6× pre→chase at 45° vs ~5.7× for all escapes). About a third of escapes are
high-turn; the rest are forward dashes.

**Caveat, stated plainly:** at 100 fps `turn_deg` is a *net heading change over the bout*, not a
resolved C-start bend (a C-start is 1.5–2 frames). So the tier is a coarse "turn-away vs forward
dash" split, **not** a kinematic C-start classifier. Read the tiers accordingly, and lean on the
directedness/proximity/threshold-sweep evidence — not the word "escape" or the tier label — when
the distinction carries weight.

**Why a threshold and not a cluster.** K-means on 322,781 pooled bouts (K=4, on turn / speed /
displacement / duration / tortuosity) reported the escape-like cluster rising from 0.167 to
0.198 during the chase — **p=0.057, a null**. Escapes are **1.45%** of bouts; k-means minimizes
within-cluster variance and will never spend a centroid on 1.5% of the mass, so escapes were
absorbed into a generic high-angle-turn cluster whose *rate* barely moves. A **7.8× effect
diluted to nothing.** Thresholding found it immediately.

Unsupervised repertoire discovery is for when you do not know what you are looking for. It is
the wrong instrument for a rare, nameable, high-amplitude class.

## Groups

| group | axes | contents |
|---|---|---|
| `config/` | — | threshold, windows, dropout tolerance |
| `epochs/` | epoch | inherited verbatim from the parent (already settle-trimmed) |
| `references/` | reference | inherited verbatim: objects, virtual twins, dish centre |
| `events/` | escape_event (× reference) | one row per escape: `bout_id`, `epoch_index`, `start_frame`, `peak_speed_mm_s`, `turn_deg`, **`is_high_turn`**, `trace_usable`; per reference `distance_at_onset_mm`, `gain_mm`, `recapture_mm`, `net_mm` |
| `rates/` | epoch | `escape_count`, **`high_turn_escape_count`**, `epoch_duration_s`, `valid_duration_s`, `tracking_dropout_fraction`, `escape_rate_per_min`, **`escape_rate_per_valid_min`**, **`high_turn_escape_rate_per_valid_min`**, **`high_turn_fraction`**, `escape_bout_fraction` |
| `trigger/` | epoch × reference | `escape_onset_distance_mm`, `ordinary_onset_distance_mm`, **`proximity_shift_mm`** |
| `pursuit/` | epoch × reference | **`gain_mm`**, **`recapture_mm`**, `net_mm`, `event_count` |
| `traces/` | epoch × reference × time | `time_s`, `delta_distance_mm` (median, baseline-subtracted at onset), `event_count` |
| `trials/` | chase_trial | `trial_id`, `ordinal`, `start_frame`, `end_frame`, `trigger_frame`, `trigger_distance_mm`, `escape_count`, `any_escape`, `valid_s`, `dropout_fraction`, **`escape_rate_per_valid_s`**, **`wall_distance_at_trigger_mm`**, `first_escape_latency_s` |
| `threshold_sweep/` | threshold × epoch | `escape_count`, `rate_per_valid_min` |

## The trial axis, and the confound that comes with it

The chase epoch is **not 180 s of continuous chasing**. It is ~12 experimenter-initiated trials
of ~5 s, delimited by the controller's `chase_trial_id`. Segmentation is reused from
`chaser_escape_freeze` (`_controller_trial_segments`) — not reimplemented.

**The escape response collapses after the first one or two trials** while freezing rises. This
is invisible when the epoch is averaged; it exists only per trial. Clean trials (<5% dropout),
per fish:

| | trials 1–2 | trials 5+ | p |
|---|---|---|---|
| escapes / valid second | **0.48** | **0.11** | **0.0002** (20/26 fish) |
| freeze fraction | 0.45 | 0.65 | **4e-05** |
| P(escape in trial) | 0.72 → 0.35 | ~0.10 | |
| **wall distance at trigger** | **9.6 mm** | **~2.0 mm** | |

Escape latency is **+0.24 s after** the proximity trigger (88% fire after it) — the chaser
closes, *then* the fish flees.

### `wall_distance_at_trigger_mm` — read this before interpreting habituation

The fish **moves to the wall after trial 1 and stays**, and at the wall it seldom escapes:
P(escape) = **0.10** at the wall versus **0.60** off it (p<0.0001). So "the escape habituated"
and "the fish is cornered with nowhere to flee" are **confounded by construction**.

Two checks rule out the geometric trap:

- **With no chaser at all** (pre epoch), being at the wall does **not** suppress fast bouts:
  3.72/min at the wall vs 3.16/min off it, **p=0.40**. There is no kinematic reason a
  wall-adjacent fish cannot produce a fast bout.
- **On trial 1**, fish that start *already at the wall* escape just as often as fish that do
  not: **0.82 (n=11) vs 0.71 (n=21), p=0.53**. A fish at the wall escapes fine — when it still
  wants to.

So the wall does not *prevent* escape. Wall position, freezing, and escape failure are three
faces of **one defensive state switch**, not three findings.

> **Do NOT regress wall distance out.** It is **downstream of the chase** — a mediator of the
> response, not a nuisance covariate. Controlling for it is conditioning on a post-treatment
> variable. It is stored so the question can be *asked*, not silently absorbed.

The direct test (does the decline survive when the fish is off the wall?) is **underpowered to
the point of uninformative**: only 47 off-wall clean trials cohort-wide, and just **3 fish**
have both early and late off-wall trials. That is absence of evidence, not evidence of absence.

**Use `escape_rate_per_valid_s`, not `escape_count`.** Trials differ in how much of the fish was
tracked, and dropout rises when the fish freezes — which is the very thing being measured.

## The three things that are easy to get wrong

### 1. `escape_rate_per_valid_min`, not `escape_rate_per_min`

Detector recall **collapses on frozen fish** — 11/32 recordings lose >10% of the chase epoch,
one loses 87.6%. A wall-clock rate would report a fish that vanished for 80 s as having escaped
less. The denominator is validly-tracked frames / fps. The wall-clock rate is written too, for
comparison, and is not the readout.

### 2. `gain_mm` and `recapture_mm` are the readout. `net_mm` is a diagnostic and it lies.

Escape (+8 mm) and recapture (−5 mm) have **opposite signs and nearly cancel**. On the real
cohort the net distance change from onset to +4 s is **+1.55 mm, p=0.55** — a null that means
nothing, because the statistic cannot see either effect. Decomposed:

| | value | p |
|---|---|---|
| gain (onset → +0.5 s) | **+8.41 mm** | **3.6e-10** |
| recapture (+0.5 s → +4 s) | **+5.45 mm** (24/32 fish) | **0.029** |
| net (onset → +4 s) | +1.55 mm | 0.55 |

`test_net_change_is_near_zero_while_gain_and_recapture_are_both_large` scripts exactly this and
fails any implementation that reports only the net.

### 3. The control that makes the pursuit claim mean anything

A fast bout is **not** an escape. The same fish making the same bout against the **same object
when it is not chasing** — the post epoch — gains nothing (+0.32 mm, p=0.61). Only against the
chaser is it directed. So: **compare `gain_mm` and `recapture_mm` across epochs for the same
reference.** That contrast is within-fish and within-object.

The virtual references give a second, within-epoch control: a fish cannot flee something that
is not there, so an object trace matched by its virtual twins is wall geometry.

## Boundaries and exclusions

- A trace window must lie **wholly inside its own epoch**. Between epochs the objects are
  repositioned; a trace crossing the boundary would splice that teleport into the distance
  curve and read as a giant spurious recapture. Excluded events are still **counted in the
  rate** — dropping them from both would understate escapes near epoch edges.
- Windows with more than `max_window_dropout_fraction` (default 0.25) untracked frames are
  excluded from traces. Also still counted in the rate.
- The count of trace-less events is reported in `qc_warnings`, never silently absorbed.

## Pseudoreplication

`events/` holds one row per escape and one fish contributes many — 2,097 events from 32 fish
are **not** 2,097 independent samples. `pursuit/` and `traces/` are already reduced to a
per-recording median and are the safe things to aggregate across recordings. Do not pool
`events/` and t-test it.

## Measurement ceiling

At 100 fps a larval C-start (~15–20 ms) spans **1.5–2 frames**, so centroid-differenced
`peak_speed_mm_s` **under-reads** true escape velocity. Nothing in this cohort exceeds
~300 mm/s where the literature reports 100–500 mm/s. **Absolute values are not comparable to
high-speed-imaging work.** The contrasts (epoch, reference) are unaffected — the same
under-reading applies to every condition — and `threshold_sweep/` exists so this can be checked
rather than assumed. On the cohort the rate ratio holds at 7.8× (100 mm/s), 16.0× (150), and
11.7× (200); at 200 mm/s the median fish shows 3 escapes in the chase.

## Relationship to `chaser_escape_freeze`

`chaser_escape_freeze` is a **trial-locked diagnostic canary**: it segments by `chase_trial_id`,
picks a proximity trigger, and classifies escape-vs-freeze by a path-length threshold over the
trial. It has no epoch baseline and no static-object control, and is marked
`classification_locked: False`.

This component is **epoch-comparative and event-based**. They answer different questions and
neither supersedes the other. If they ever disagree about whether a fish escaped, the
difference is the definition (path length over a trial vs peak speed of a bout), not a bug.

## Results

`docs/diagnostics/goodcopbadcop_escape_pursuit_2026-07-14.md`.
