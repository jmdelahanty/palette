# Validated-behavior distribution successor — 2026-09-02

## Status

Implementation checklist for a compact, immutable distribution companion to
one exact validated-behavior cohort publication. The first target is the
GoodBatBadBat Phase-C publication, but every computation is selected through
generic metric and scope contracts. No recording IDs, chaser colors, chaser
roles, or cohort-specific scientific formulas may be hard-coded.

## Decision

The large validated-behavior export remains the lossless cohort query surface.
The distribution successor adds the small facts that are missing for repeated
distribution work:

1. one recording-keyed bout-observation table that joins canonical bout
   kinematics to exact protocol-semantic epoch membership;
2. one recording-keyed inter-bout-interval table, including intervals that
   cross epoch boundaries for the whole-session view;
3. fixed-width, shared-edge bins reduced independently inside each recording;
4. equal-recording-weight cohort summaries, alongside a separately labelled
   pooled-observation diagnostic; and
5. one backend-neutral visualization payload consumed by static Matplotlib and
   interactive Plotly renderers.

The successor binds the exact parent export manifest, bundle-set manifest,
targeted epoch-child receipts, metric registry, resolved bin edges, output
schemas, and every output-file digest. It does not mutate recording Zarrs,
rebuild Phase C, activate a selector, or claim production authority.

## Scientific grains and weighting

The distribution axis and the observation-weighting axis are independent.

| Weighting | Eligible source | Meaning |
|---|---|---|
| `event` | bouts and inter-bout intervals | each finite event contributes one |
| `frame` | provider motion and fish--chaser distance | each valid acquisition-frame observation contributes one |
| `time` | provider motion and fish--chaser distance | each valid observation contributes its sealed elapsed time |

`frame` and `time` are not aliases. Frame weighting answers “what fraction of
valid sampled frames?”, while time weighting answers “what fraction of valid
observed time?”. Their equality is expected only under perfectly uniform
sampling without gaps. The exact elapsed-time field and validity rule are part
of every metric recipe. A time-weighted transition contributes to an epoch
only when both of its acquisition-frame endpoints lie in that same half-open
semantic interval. Transitions across an epoch boundary are retained for the
whole-session view but cannot leak elapsed time into either epoch.

Bout and interval histograms remain event weighted. Event support additionally
records a rate per valid tracked minute, using the exact provider-motion time
support for the same recording and scope. Frames or bouts are never treated as
independent cohort replicates: cohort curves first normalize within a
`recording_id`, then average finite recording fractions with equal recording
weight.

## Scope contract

Every metric is reduced for these ordered scopes:

1. `whole_session` — every valid row/event in the selected recording source;
2. `chaser_pre` — exact half-open semantic pre interval;
3. `chaser_training` — exact half-open semantic training interval; and
4. `chaser_post` — exact half-open semantic post interval.

Whole-session inter-bout intervals are derived between every adjacent pair of
canonical bouts and therefore retain cross-epoch intervals. An interval is an
epoch member only when the previous bout end and next bout start both lie
inside that same exact epoch. A cross-epoch interval is valid whole-session
evidence but is not assigned to either neighboring epoch.

Canonical bout rows obtain epoch membership only from the sealed
`per_epoch_bouts/bout_source_row` mapping. They are never assigned from an
approximate time, row order, label parse, or inferred trial boundary.

## Metric registry v1

The first registry contains:

- bout duration, path length, net displacement, mean speed, peak speed, and
  tortuosity;
- signed net bout heading change, absolute net bout heading change, and
  within-bout heading path;
- inter-bout interval;
- filtered and smoothed physical speed;
- smoothed frame-to-frame path distance;
- signed smoothed heading change, signed smoothed angular velocity, and
  smoothed angular speed; and
- fish--chaser physical distance, grouped by exact position-provider and
  semantic chaser role.

Signed angular metrics use symmetric ranges. Anatomical angle differences use
the closed `[-180, 180]` contract. Ordinary nonnegative ranges start at zero.
Open ranges resolve once across all exact valid parent rows by rounding the
observed maximum up to the declared bin width; the resolved edges and source
audit are then immutable. Tortuosity uses shared log-spaced positive bins
because near-zero displacement can yield scientifically valid values spanning
several orders of magnitude; this retains the full tail without compressing
the main distribution or silently clipping outliers. A guard rejects an
implausibly large bin roster.

Peak-to-peak bout heading is not in the current sealed canonical or epoch-bout
surface and is not fabricated in v1. Adding it requires a separately specified
frame-to-bout reducer and a new metric-registry version.

## Heading and interval derivation

The whole-session bout-heading extension uses the same source algorithm as the
sealed epoch child: select angular-valid smoothed headings inside each
inclusive canonical bout frame interval, wrap the first-to-last difference to
`[-180, 180]`, and sum absolute wrapped adjacent differences. For every
epoch-mapped bout, the derived values must agree with the already persisted
epoch-child values before publication. A mismatch fails closed.

Inter-bout intervals preserve the producer-authored `interval_s` values from
the exact bundle-bound canonical swim-bout run. Before those values are
admitted, every interval must join the adjacent canonical bout IDs and frames
exactly, its `interval_frames` must equal
`max(0, next_start_frame - previous_end_frame)`, and its duration must agree
with that frame gap divided by the exact producer FPS within the producer's
declared numerical tolerance. The producer FPS is independently checked
against the sealed provider-motion elapsed-time axis and epoch summary. This
retains the producer's floating-point boundary convention instead of
reconstructing a subtly different value at a histogram edge. Per-epoch
denominators and bin counts are then checked exactly against the
receipt-targeted persisted interval histograms.

## Visualization contract

Each distribution figure has aligned Whole, Pre, Training, and Post panels
with shared bin edges. The default curve is the mean of recording-normalized
fractions. A pooled-observation curve may be selected only under the explicit
label “pooled observations”; it is diagnostic and cannot replace the default.
The payload exposes denominators, excluded rows, contributing-recording counts,
units, bin width, weighting, validity policy, parent manifest digest, and recipe
digest.

The interactive trace view reads the same exact Phase-C provider-motion rows
on demand for one selected recording. Its x coordinate can switch between
`acquisition_frame_id` and `time_s`; this is a display-coordinate choice, not
two scientific payloads. Bounded display decimation, when needed, is recorded
in the trace payload and never changes the persisted distribution results.

Chaser roles and appearance are read from the Phase-C occurrence dimension.
Semantic role controls marker shape/text; protocol-authored color remains an
independent appearance channel. Neither renderer infers aggression from a dot
color or hard-codes an identity-to-role mapping.

## Implementation checklist

### Contract and source closure

- [x] Define and validate the v1 metric/axis/weighting registry.
- [x] Strictly open one exact parent export; reject `latest` discovery.
- [x] Bind the parent manifest, export plan, analysis-unit policy, membership,
  and bundle-set identities.
- [x] Load only the declared epoch arrays through each exact child receipt.
- [x] Preserve all parent recordings in support denominators, including typed
  noncontributors.

### Observation materialization

- [x] Materialize canonical bout observations with exact epoch membership.
- [x] Recompute whole-session bout heading from exact angular-valid motion and
  prove equality to persisted epoch values.
- [x] Materialize whole-session and within-epoch IBI observations.
- [x] Validate epoch IBI counts against persisted receipt-bound histograms.
- [x] Persist source row keys and child/source digests needed for independent
  reuse.

### Reduction and publication

- [x] Resolve one shared edge roster per metric and persist its digest.
- [x] Reduce event-, frame-, and time-weighted bins within each recording.
- [x] Persist exact candidate, valid, excluded, denominator, and rate support.
- [x] Compute equal-recording summaries and separately labelled pooled values.
- [x] Atomically publish exact Arrow schemas, file hashes, and a self-digested
  manifest; refuse overwrite and unrecorded files.
- [x] Add a strict reader that rechecks identities and denominator arithmetic.

### Shared visualization

- [x] Build one self-digested backend-neutral distribution payload.
- [x] Add static four-scope histogram figures and an atomic HTML report.
- [x] Add a modular Marimo/Plotly explorer with metric, weighting,
  provider/role, and cohort-statistic controls.
- [x] Add a bounded one-recording trace view with frame/time x-axis choice.
- [x] Carry exact protocol appearance independently from semantic role styling.

### Verification and rollout

- [x] Unit-test registry rejection, scope assignment, angle wrapping, IBI
  membership, time weights, empty denominators, equal-recording versus pooled
  summaries, manifest tamper detection, and both renderers.
- [x] Run focused tests and `marimo check` outside the Codex sandbox.
- [ ] Run required repository CI and retain every required green result before
  integration.
- [ ] Publish one immutable Phase-C-bound canary and visually inspect all
  registered metric families.
- [ ] Record paths, row counts, digests, runtime, and any typed exclusions in
  this document without rewriting prior evidence.

## Canary evidence

The first real-recording canary used
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` from the exact Phase-C parent.
The final pre-cohort canary produced 1,445 bout observations, 1,444 exact
producer-authored interval observations, 120 support strata, 5,555 nonzero
recording bins, and 14,836 full cohort-axis bins. Its temporary distribution
record digest was
`36b009b1c0a1646a05fc89b8886aff9bee8b09f60e5961ce1d1d1cb325592800`;
the 24-figure HTML report digest was
`b71c7c7280c934040ad96c9693a6bf8e3ecdcebfdd0075abddab37d5b85d466d`.
These `/tmp` outputs are validation specimens, not the final cohort
publication or selector authority.

The canary exposed two conditions that now fail closed or have an explicit
contract: producer-authored IBI floats must be retained at exact histogram
boundaries, and high tortuosity from near-zero net displacement uses a
log-spaced positive axis rather than silent clipping. Time-weighted epoch
histograms also require both transition endpoints inside the same epoch;
boundary-crossing transitions contribute only to the whole-session view.

## Deferred work

- A recording-Zarr-local histogram cache is optional acceleration, not a new
  authority. If added, it must bind the same exact observation source and
  recipe; exact unbinned recording data remains authoritative.
- Distributed shard fanout may follow after profiling the serial canary. It
  must preserve whole non-overlapping output shards and deterministic fan-in.
- Peak-to-peak bout heading and other new scientific metrics require explicit
  successor contracts rather than viewer-side formulas.
