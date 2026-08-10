# Chaser analytics roadmap — proposed components and modeling decisions

<!-- contract-meta
version: 1
status: draft
last_verified: 2026-08-10
implementation: specified-only
-->

Source: design discussion 2026-08-10, following the five-lens review
(`design_review_findings_2026-08-09.md`). This is a roadmap/todo, not a contract.
Nothing here is implemented. Each item becomes real via the component lifecycle
recipe at the bottom.

## Context: the existing suite, as classes

The 13 declared components in `chaser_behavior_full_v2.yaml` form five classes:
spatial occupancy (quadrant/radial/near-field/detection), distance-conditioned
state marginals (response regimes, escape/freeze summary), event detection and
rates (escape events, bout response), egocentric/orientation (bearing, gaze),
and epoch summaries + group statistics. Strengths: null-control discipline
(virtual twins, dish-center controls, annulus normalization, valid-time
denominators) and the declared DAG. Characteristic gap: almost everything is a
time-marginal; sequence/dynamics are unexplored except inside escape events —
which is the one component that produced the strongest result.

## Proposed components, ranked by expected scientific payoff

**A0 — Threat-aligned frame table (shared base; build first).**
Per valid-tracked frame: chaser distance, radial approach velocity (already
computed in response regimes), egocentric bearing, state label
(immobile/swim/escape from existing definitions), event flags (escape onset,
recapture), trial index, and explicit censoring flags (track-invalid,
transition-invalid). A1–A3 consume this. This is where the censoring policy is
implemented ONCE: invalid frames leave the risk set; track loss during
immobility is censoring, NOT state exit (the tracker preferentially loses
frozen fish — 80.5 s hole precedent).

**A1 — Escape hazard model (highest priority).**
Discrete-time GLM over the frame table: P(escape onset) ~ distance +
approach velocity + bearing + time-since-last-escape + trial. Replaces the
descriptive threshold sweep with interpretable trigger geometry (distance per
se vs. looming/approach-rate), unifies proximity triggering and habituation in
one model, and handles freeze-correlated dropout via the risk set. Needs
`statsmodels` (NOT currently in pyproject deps — pin deliberately; sklearn has
no SEs/CIs).

**A2 — State-transition kinetics.**
Semi-Markov treatment of immobile/swim/escape: dwell-time distributions and
transition hazards conditioned on chaser distance. The kinetics of freezing is
the most important unmeasured thing in the dataset given the
"response is freeze, not approach" finding. Must report censored fraction as a
first-class output.

**A3 — Escape directionality and destination.**
Heading change at escape onset relative to chaser bearing (directed flight vs
undirected startle); escape endpoint location (toward wall?). Connects the
escape story to the wall-proximity-as-chase-mediator finding. Reuses
egocentric bearing + virtual controls.

**A4 — Response latency / anticipation.**
Cross-correlation of chaser approach vs fish acceleration per fish per trial;
lag distribution and lag-shift across trials as a learning readout (more
sensitive than the occupancy readouts that died as artifacts). Clean because
the chaser is programmatic. 100 fps = 10 ms quantization; fine for 50–500 ms
lags.

**A5 — Individual response phenotypes.**
Random-slopes/ICC repeatability across recordings; cluster fish by response
profile (escape-dominant vs freeze-dominant). Motivated by 12/32 fish moving
opposite the group mean on thigmotaxis. The biological subject is the repeated
experimental unit. An explicitly assigned acquisition batch may be used as an
optional nuisance/blocking variable; neither timestamps nor recording names may
stand in for that identity.

**A6 — RANDOM_NON_CHASING contrast axis.**
`chaser_behavior.py` already defines AGGRESSIVE / RANDOM_NON_CHASING / INERT.
Only the quadrant endpoint has a specificity contrast today. Add a generic
"compute component X restricted to behavior-class Y" axis on the profile so
every metric gets a within-paradigm null (real visual stimulation, no pursuit
contingency) — stronger than virtual-twin rotation. Mostly plumbing, no new
science code.

**A7 — Escape-triggered body shape (opportunistic).**
Midline snippets from `subject_shape_runs` at escape onsets: turn-angle and
curvature classification. Caveats already documented: 1/40 coverage, unvetted
tails, shape-not-kinematics, 100 fps under-reads C-start peaks — classify turn
geometry, not speed.

## Resolved prerequisites and remaining gates

- W1.1 (speed-source contract), W1.2 (arena-geometry notes + hard-fail virtual
  controls), and W1.4 (registry-derived `subject_id` plus optional
  `acquisition_batch_id`) are implemented on the design-review integration
  branch. They must be integrated and remain green before any new component
  begins.
- Add `statsmodels` to pyproject deliberately before A1.

## Component lifecycle recipe (the "palette way" — what each item above needs)

1. Design/contract doc first (`docs/`, contract-meta, `implementation:
   specified-only`): question, inputs, outputs with units, denominators named,
   null model, caveats. Science decisions pre-registered here — for A1: escape
   onset definition, covariate set, time bin, censoring policy.
2. Schema module declaring every output array with units/axes (pattern:
   `chaser_distance_base_schema.py`; prefer versioned `ArrayContract` per
   fix-queue W4.4).
3. Sealed upstream reads via snapshot readers (`chaser_distance_io.py`
   sealed-constructor + `require_derived_surface_authority`); never raw zarr
   paths.
4. Pure compute core — arrays in, arrays out, no I/O.
5. Null machinery: virtual twins where geometry allows + A6 behavior-class
   contrast.
6. Publication through the existing component writer / atomic publisher —
   tombstones, completion markers, `run_provenance` come free at the
   finalization gate.
7. Profile YAML entry with `depends_on`.
8. Tests in the house idiom: closed-form synthetic recovery (e.g., Poisson
   process with known distance-dependent rate must recover its coefficients;
   hand-built freeze bouts must yield the constructed dwell distribution) plus
   guardrail tests (valid-time denominators, no raw speed, geometry-notes
   propagation, censoring under simulated track loss).
9. Export + stats wiring: Arrow columns/units in `analytics_exports/contracts.py`,
   registry-derived subject identity, optional acquisition-batch provenance,
   and `MetricSpec` with an honest primary/exploratory flag and a real FDR
   family.

Effort shape: steps 2–3, 6–9 are mechanical (agent-friendly); step 1 is the
human work (~a day of thinking per family). A0 ≈ one week with tests; each of
A1–A3 a few days on top of A0.

## Decision record: no ARHMM/MoSeq now — and the trigger for revisiting

Decision (2026-08-10): do NOT reach for unsupervised latent-state models
(ARHMM/MoSeq/keypoint-MoSeq) for the chaser questions. Reasons:
- Observables are centroid + heading; discoverable states ≈ speed/turn regimes
  we already define by hand. The model cannot out-discover its inputs.
- The questions are covariate-driven (escape ~ distance/approach rate); vanilla
  ARHMM has no input dependence — the chaser doesn't exist in the model.
- The target event is rare (escapes = 1.45% of bouts); unsupervised clustering
  absorbs rare states into bulk clusters (k-means precedent in this dataset).
- Latent-state fits carry heavy provenance/interpretability debt (seed,
  hyperparams, state count; labels change across refits) and don't admit the
  house closed-form test idiom.

Revisit triggers (either one):
1. A1's residuals show STRUCTURED unexplained switching — same fish, same
   chaser covariates, alternating responsive/unresponsive on minutes
   timescales. That is the empirical signature of a latent internal state and
   justifies a GLM-HMM (input-driven, Ashwood-style) where a slow hidden state
   modulates the stimulus→response mapping.
2. Dense, vetted pose (high-coverage keypoints/tail) lands AND the question
   becomes repertoire discovery ("did training restructure fine-grained swim
   syllables in unhypothesized ways?") across many fish.

Rule of thumb this encodes: unsupervised state models are discovery
instruments for rich observables and unknown repertoires; confirmatory
questions about known states, rare events, and explicit covariates want
defined-state models (GLM hazard, semi-Markov). Let the simple model's
residuals tell you when you've earned the complex one.

---

## Chaser stimulus schedule — effective-model analysis and redesign

Observed (2026-08-10): fish either hold thigmotaxis through whole sessions or
freeze after 2-3 chaser presentations, even after "lowering" attack
probability to 10%/s with a post-chase cooldown.

### The effective model of 10%/s is near-continuous threat

Per-second Bernoulli hazard compounds: P(chase within 10 s) = 1 − 0.9^10 ≈ 65%;
within 30 s ≈ 96%; expected chases in 180 s ≈ 18 without cooldown, ~8-12 with a
10-15 s cooldown. Mean quiet interval ~10 s, memoryless — no interval is safe,
no timing learnable, and tonic arousal cannot decay between chases. The session
is one continuous threat state; measurements reflect tonic defensive state, not
phasic responses to discrete events.

### Why freezing + ceiling thigmotaxis are the expected outcome

1. Density (above).
2. Uncontrollability — own data: escapes gain distance, chaser recaptures,
   net ≈ 0. Active escape does not work from the fish's perspective;
   inescapable threat drives the shift from active (flight) to passive
   (freezing) defense. Fish freezing after 2-3 presentations are learning
   fast, not failing to learn.
3. Unpredictability of a memoryless schedule pushes tonic anxiety-like state
   (ceiling thigmotaxis) → no dynamic range left for learning readouts
   (consistent with spatial-avoidance readouts dying, habituation
   underpowered).

Bimodal presentation (thigmotaxis-lockers vs early-freezers) matches the
proactive/reactive coping-style axis → this is A5's dependent variable, not a
nuisance. Practical cost: the tracker preferentially loses frozen fish, so the
paradigm currently pushes fish into the least-measurable state.

### Design levers

- Set rate by desired chase count: N ≈ T / (C + 1/p). For 3-5 chases in 180 s
  with recovery: p ≈ 1-2%/s with 20-30 s cooldown — or schedule chases
  directly (below).
- Controllability is the biggest lever. For avoidance LEARNING the fish needs
  an action that works: chase terminates on reaching a region / exceeding an
  escape criterion / duration cap; strongest = signaled avoidance (1-2 s
  warning cue). Unsignaled + uncontrollable ⇒ freezing, by design. If the goal
  is defensive-state dynamics instead, current design is fine and freezing is
  the dependent variable.
- Add a pre-chaser baseline epoch per session (dissociates contextual/
  carryover fear — consistent cross-session thigmotaxis suggests the dish is
  becoming threat-associated).

### Schedule design: controlled count WITH unpredictability

Predictability lives in the hazard function shape, not in "randomness" per se:
fixed 30 s interval = fully predictable (hazard spike); uniform jitter =
partially predictable (hazard rises toward the window end); constant hazard
(exponential ISI) = memoryless = maximally unpredictable. Options that keep
uncertainty while controlling dose:

1. **Fixed-N random placement with minimum gap (recommended).** Draw exactly N
   chase onset times uniformly at random over the session subject to a
   minimum spacing Δ (refractory for recovery). Guarantees identical threat
   dose per fish/session (removes chase-count as a between-subject nuisance —
   matters for A5 and all epoch summaries), guarantees recovery windows,
   timing not practically anticipatable at small N.
   Caveat: with a min gap and fixed session length the hazard is not perfectly
   flat — late in a session with no chases yet, an ideal observer can infer
   one is coming. Negligible for a fish at N≈4 over 180 s; becomes real only
   at large N or long min gaps. Fix if needed: draw N itself from a small
   range (3-5) so even the count is uncertain.
2. **Shifted/truncated exponential ISIs** (ISI = C + Exp(μ), capped): purest
   memorylessness (flat hazard after refractory — ambush-predator-like);
   count varies Poisson-like across fish (dose no longer matched, variance
   goes straight into between-fish noise). At current n, trading a little
   memoryless purity for matched dose (option 1) is the better deal; use this
   only when memoryless purity matters more than dose matching.
3. Keep the current dense regime as a named protocol condition ("tonic
   threat") rather than deleting it — dense/uncontrollable vs sparse/
   controlled-count is itself an interesting axis, and cohort data already
   exists in the dense regime. The WITHIN-FISH contrast is arguably a better
   experiment than either regime alone: does the same fish freeze under tonic
   threat but show active avoidance under sparse, escapable threat? Record
   the regime in protocol metadata (protocol_hash / stimulus_protocols) so
   analyses can stratify, and log the realized chase times + RNG seed into
   stimulus provenance.

### Tuning loop (couples to A0-A2)

Time-since-last-chase is the natural covariate of the A1 hazard model, and A2
gives freeze-recovery curves per ISI. Pilot a few fish at ~4 chases/session
with ≥30 s gaps; the recovery curve then SETS the final ISI: correct spacing
is the spacing at which freeze probability visibly returns toward baseline
between chases. If it doesn't return, the session is still saturated — adjust
by curve, not by feel.
