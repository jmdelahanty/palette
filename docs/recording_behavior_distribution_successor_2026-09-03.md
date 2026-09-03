# Recording behavior distribution successor — 2026-09-03

## Status

Implemented on the development branch as one immutable, recording-local
distribution component reusable by static figures and Marimo. The first live
adapter consumes the validated GoodBatBadBat recording bundle, but the
component is deliberately not a chaser product. Required CI and live canary
publication remain outstanding, so this branch is not yet merge-ready or a
production authority.

## Decision

Publish distributions as the composition of three independent contracts:

1. a **metric provider** names a validated value, its scientific grain, units,
   validity, weighting, grouping dimensions, and exact source binding;
2. a **scope provider** names reproducible row membership without changing the
   metric; and
3. a **histogram recipe** fixes the grid, endpoint convention, normalization,
   and denominator.

The canonical unbinned arrays and event tables remain scientific authority.
The recording distribution is an immutable derived summary bound to those
sources. A plot is only a renderer over the persisted summary and may not
silently rebin, reassign membership, or change denominators.

The scientific run stores the reusable numeric distribution component, not a
renderer-specific image as authority. The shared Matplotlib function is
PNG/PDF-ready and the recording explorer renders Plotly from the same restored
series. Persisted visualization snapshots may be added as separately bound
derived artifacts without changing the distribution run.

The initial recording-local path is:

```text
analysis/recording_behavior_distribution_runs/<run_id>/
```

Runs are named, immutable, selector-ineligible publications. They do not
replace a source selector or mutate any completed scientific child.

## Why this is separate from the existing cohort product

The receipt-bound validated-behavior cohort distribution already contains
per-recording sparse bins and equal-recording cohort summaries. Its physical
carrier is the exported cohort directory, however, and its current default
scope roster is fixed to whole-session plus three chaser epochs.

Individual recording Zarrs currently contain several valid but fragmented
surfaces:

- semantic-epoch bout and inter-bout-interval histograms;
- chaser radial distance bins; and
- a historical whole-session swim-bout dashboard with a different
  recording-range/30-bin recipe.

The new component supplies one recording-local contract before cohort export.
The existing cohort artifact remains immutable. A later cohort fan-in may read
these sealed recording components and must prove results equivalent to the
current lossless-export reducer before replacing it.

## Scope providers

Every scope record carries `scope_id`, `scope_label`, `scope_family`,
`scope_provider_id`, ordering, membership policy, source evidence, and a
self-digest. Scope IDs have meaning only inside their declared provider.

Initial providers are:

| Provider | Scopes | Membership authority |
| --- | --- | --- |
| `whole_session.v1` | `whole_session` | every valid source row/event |
| `protocol_semantic_intervals.v1` | protocol-defined roles such as `chaser_pre`, `chaser_training`, `chaser_post` | exact persisted semantic membership or exact half-open frame intervals, as required by the metric grain |
| `named_session_time_brackets.v1` | caller-named brackets | canonical session timestamps mapped with explicit half-open `[start, end)` semantics and no interpolation |

Moving-grating experiments can provide grating blocks or stimulus phases.
Video-only recordings can publish only `whole_session`, or explicit named time
brackets. They never receive fabricated chaser epochs. New paradigms add a
scope adapter, not a new histogram implementation.

The first production source adapter obtains exact session time from the
validated GoodBatBadBat bundle's paired relative-frame consensus clock. This
does not make named brackets chaser-specific: a future moving-grating or
video-only bundle supplies its own validated session-timebase adapter and then
uses the unchanged scope, reducer, storage, and renderer contracts.

For a named time bracket, exploratory filtering may remain transient. A
published bracket must additionally retain requested time bounds, canonical
timebase identity, selected-row count, uncovered/invalid-row count, exact
endpoint policy, and source digest. Event metrics must declare whether their
membership is source-authored, fully contained, or anchored to an event frame;
the renderer may not choose this policy.

## Initial metrics and grains

The existing declarative validated-behavior metric registry remains the
starting registry:

- bout duration, path length, net displacement, mean speed, peak speed,
  tortuosity, signed/absolute net heading change, and heading path;
- inter-bout interval;
- filtered and smoothed speed, per-transition path distance, heading change,
  angular velocity, and angular speed; and
- optional fish--chaser distance grouped by exact position provider and
  behavior role.

Chaser distance is an optional metric adapter. It is absent, not null-filled,
for recordings without a compatible chaser-relative authority.

Frame/time weighting and event weighting remain distinct. Counts, weighted
denominators, fractions, excluded counts, valid observed duration, and event
rate support are persisted separately.

## Bin compatibility and cohort fan-in

Each recording persists sparse **canonical grid indices**, not merely a local
zero-based bin number. Fixed-width nonnegative and signed metrics therefore
share an origin and width even if their observed terminal ranges differ.
Logarithmic metrics share an exact log-grid definition. A cohort fan-in can
take the union of occupied grid indices, insert structural zeros for finite
recording denominators, and calculate equal-recording summaries without
re-reading frame rows.

The recording component also persists its observed minimum/maximum and local
resolved display extent. Those bounds are evidence, not permission to clip.
Underflow or overflow is a contract error for a fixed-range metric and can
never be hidden in the plotted denominator.

## Shared visualization surface

One renderer-neutral series projection reconstructs ordered bins from the
persisted scope, metric, group, support, and sparse-bin tables. Matplotlib and
Plotly consume that same projection. The initial figure family provides:

- a whole-session panel;
- one panel or overlaid trace per selected scope;
- probability per bin as the default y-axis;
- exact raw count and finite denominator in labels/hover; and
- explicit metric units, bin edges, weighting, source identity, and recipe
  digest.

Path length and net displacement remain distinct metrics. Chaser role and
protocol color remain independent dimensions; no role-to-color mapping is
hard-coded.

## Implementation checklist

### Contract and pure reducer

- [x] Add validated generic scope records and reject duplicate IDs, duplicate
      order, invalid interval bounds, unsupported membership policies, and
      digest mismatch.
- [x] Add deterministic whole-session, exact frame-interval, explicit
      source-membership, and named session-time bracket mask builders.
- [x] Add a recording reducer that accepts explicit metric inputs, validates
      row alignment, and emits support plus sparse canonical-grid bins.
- [x] Preserve distinct event/frame/time weighting and exact denominator
      arithmetic.
- [x] Unit-test empty scopes, invalid rows, interval boundaries, signed axes,
      log axes, grouping, and arbitrary non-chaser scope names.

### Recording adapters

- [x] Add a validated-bundle core behavior adapter for canonical bouts,
      inter-bout intervals, and provider motion.
- [x] Add the protocol-semantic scope adapter using exact chaser-epoch
      evidence; bout membership must use the persisted source-row mapping.
- [x] Add optional exact chaser-distance metrics without making them a core
      dependency.
- [x] Add named session-time brackets over the bundle-bound acquisition
      timebase with `[start, end)` semantics and no interpolation.

### Immutable Zarr publication

- [x] Plan and atomically publish one selector-ineligible run below
      `analysis/recording_behavior_distribution_runs`.
- [x] Persist scope, metric, group, source, support, and sparse-bin registries;
      bind every array digest in a self-digested run manifest.
- [x] Refuse overwrite, selector mutation, stale bundle/source bindings,
      unconsolidated final visibility, and partially published runs.
- [x] Add a consolidated-metadata reader that validates the exact requested
      run and targeted payload digests.

### Shared renderers and migration

- [x] Add one backend-neutral persisted histogram-series projection.
- [x] Add a PNG/PDF-ready static Matplotlib renderer and a Marimo/Plotly
      renderer over that same projection.
- [x] Mount whole-session and exact chaser-epoch bout, IBI, motion, and
      distance figures in the recording explorer.
- [ ] Stop producing the historical recording-range/30-bin swim-bout
      histogram after equivalent generic coverage is deployed; retain a
      read-only compatibility reader only where historical archives require it.
- [ ] Teach the cohort publisher to admit recording components as an optional
      acceleration path and prove byte/numeric equivalence before promotion.

### Validation and rollout

- [x] Run focused unit and real-Zarr tests outside the sandbox.
- [ ] Publish one commit-pinned selector-ineligible chaser canary and inspect
      whole-session plus pre/training/post views.
- [ ] Publish one video-only or non-chaser fixture and prove the same renderer
      works with no chaser capability.
- [ ] Run every required CI check before merge, shared-checkout update,
      selector activation, or production use.

### Pre-CI validation evidence

- The focused recording-distribution/source suite passes 75 tests, including
  real-Zarr atomic publication and consolidated-read round trips.
- The broader affected Marimo component suite passes 74 tests.
- `marimo check` passes for the modular recording explorer.
- Static compilation, Ruff, and `git diff --check` pass for the implemented
  files.

These checks are local development evidence. They do not replace required CI
and do not authorize merge or deployment.

## Non-goals for the first implementation

- No arbitrary viewer-side scientific rebinning.
- No inferred protocol roles from names, colors, row order, or durations.
- No flattening all scientific grains into one table.
- No requirement that every recording provide every optional metric family.
- No rewrite of the already sealed cohort distribution publication.
