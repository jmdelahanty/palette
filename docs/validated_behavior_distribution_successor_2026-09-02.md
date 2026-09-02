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
- [x] Run required repository CI and retain every required green result before
  integration.
- [x] Publish one immutable Phase-C-bound canary and visually inspect all
  registered metric families.
- [x] Record paths, row counts, digests, runtime, and any typed exclusions in
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

## Full cohort evidence

The commit-pinned cohort publication ran from Palette commit
`0f223d30f32fabf333d3d07f1f0c23f82b739a23` against Phase-C export
`goodbatbadbat-validated-behavior-phase-c-20260902-19a006cc`, parent manifest
`8fb2c7ecabeff2b13b6178416842f477d99c55ae8d7df540f5dd71eea7ad1646`.
The immutable distribution is
`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_distributions_20260902_0f223d30/distribution`.
Its record digest is
`bb90c675747fdc3403de24a21bd824a8603526b8d529f5d795ed13c571fe46e1`.

The publication retained all 84 parent members: 80 recordings contributed and
the four exact members with invalid semantic selection remain typed
noncontributors in every applicable denominator. It contains 101,233 bout
observations, 101,153 producer-authored inter-bout intervals, 9,600
recording-support rows, 475,258 nonzero recording-bin rows, and 71,992 complete
cohort-bin rows across 17 metric specifications. Serial wall time was 12m04s;
peak resident memory was 7,830,316 KiB (about 7.47 GiB), with no swap. The
strict reader independently reopened the final generation and enumerated all
17 metric contracts.

The first immutable full-evidence report is
`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_distributions_20260902_0f223d30/report`.
It contains 24 PNGs plus an HTML index, has v1 record digest
`3de2dab6c426b3c097fd806f65428440dfa93724077533b37d79e8cd78823100`,
and rendered in 16.39 s with 1,203,900 KiB peak resident memory. Visual review
covered bout kinematics, bout heading, interval, motion-speed,
motion-displacement, motion-heading, and chaser-distance families in both
available frame/time weightings.

That review found sparse but valid tails extending the full linear axes: for
example, mean-recording mass above 100 mm/s was about 0.052% for filtered
speed, above 1,000 deg/s was about 0.370% for absolute signed angular-velocity
tails, and above 60 s was about 0.097% for inter-bout intervals. The complete
histograms remain authoritative; no observation was clipped or reclassified.
Viewer commit `957f2e2b25ebddeae1f59e3c9f9a069db5aba2cf` adds a display-only
`central_99` choice that uses whole persisted bins and chooses one shared range
retaining at least 99% of the equal-recording mass in every displayed
scope/group series. `full_evidence` remains an immediate choice. Every range
record carries its bounds, retained/omitted fractions, reason, and digest;
schema-v2 report validation rejects internally inconsistent claims while the
reader retains v1 report compatibility.

The immutable central-view report is
`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_validated_behavior_distributions_20260902_0f223d30/report_central99_957f2e2b`.
Its v2 record digest is
`b19c17b8a8845572f5028b3b6eaba9141e0f5645b7e2c1bba9745fbeb22fc683`;
24 figures plus HTML rendered in 20.30 s with 1,201,780 KiB peak resident
memory. Representative resolved central bounds were 45.6 s for IBI (minimum
99.003% retained), 37 mm/s for filtered speed (99.012%), ±700 deg/s for
signed angular velocity (99.017%), 440 deg for within-bout heading path
(99.017%), and 75 mm for chaser distance (99.304%). Tortuosity retains its
complete 0.00126–7,943 logarithmic axis. Both immutable report generations
strictly reopened with the v2 reader. The Plotly/Marimo backend rendered all 48
full/central variants from the exact cohort payload; 23 central ranges applied
and tortuosity correctly retained its complete logarithmic range.

Focused distribution validation passed 15 tests; the neighboring cohort,
export, group-statistics, and rendering suite passed 74 tests. Marimo's
structural checker, Python compilation, formatting, and whitespace checks also
passed.

Required repository CI run `33676511597` completed successfully for exact PR
head `bec43b9a607c09f7f7981f9824d91304d868ee0f`: all seven contract,
packaging, and collection gates and all 16 non-GPU test shards passed (23
required checks total). PR 115 then merged through the repository's merge-commit
workflow on 2026-09-02 at `0440c248ae45c124d80de37698b8a2f968f1a65a`.
The remote `main` ref was independently verified at that exact merge commit.
Integration did not activate a selector, mutate a registry or source Zarr,
deploy code, or move the shared `/groups` checkout.

## Post-merge explorer review

The merged reader reopened the exact full-cohort distribution at record digest
`bb90c675747fdc3403de24a21bd824a8603526b8d529f5d795ed13c571fe46e1`
and enumerated all 17 registered metrics. The read-only Marimo explorer started
successfully from merge commit `0440c248ae45c124d80de37698b8a2f968f1a65a`,
and its HTTP application endpoint responded successfully on localhost.

Visual review covered all 24 PNGs in the immutable central-range report bound
by record digest
`b19c17b8a8845572f5028b3b6eaba9141e0f5645b7e2c1bba9745fbeb22fc683`.
The event, frame, and time labels are explicit; Whole, Pre, Training, and Post
panels share their contracted axes and report 80 contributing recordings;
signed distributions preserve both directions; tortuosity retains its complete
logarithmic evidence range; and chaser distance distinguishes semantic role by
glyph and position provider by line style while retaining the cohort's exact
black protocol appearance. Frame- and time-weighted curves are visually almost
identical for this nearly uniform-sampling cohort, as expected, but remain
separate selectable contracts. No blocking layout, clipping, missing-panel, or
role/provider-encoding defect was observed.

The 10-degree signed frame-to-frame heading-change recipe places most central
mass in its single `[-5, 5]` degree bin. That is faithful to the sealed v1 bin
recipe rather than a renderer defect, but a finer scientific view would require
a versioned metric recipe and a new distribution successor; a viewer must not
silently re-bin it.

## Deferred work

- A recording-Zarr-local histogram cache is optional acceleration, not a new
  authority. If added, it must bind the same exact observation source and
  recipe; exact unbinned recording data remains authoritative.
- Distributed shard fanout may follow after profiling the serial canary. It
  must preserve whole non-overlapping output shards and deterministic fan-in.
- The valid serial full-cohort reducer peaked at about 7.47 GiB. A later
  optimization should stream or shard recording histograms before deterministic
  fan-in and prove byte-identical cohort/support outputs; this is a performance
  successor, not permission to weaken validation or rewrite the sealed cohort.
- Peak-to-peak bout heading and other new scientific metrics require explicit
  successor contracts rather than viewer-side formulas.
