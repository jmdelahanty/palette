# Brief: chaser schedule importer

Ingest `fixed_n_min_gap_v1` planned schedules and realized chase events.

**Date:** 2026-08-10
**Status:** specified-only; implementation is blocked on obtaining and inspecting
the Citrus H5 contract fixture
**Production effect:** none; no selector, registry authority, profile, or
canonical-data changes are authorized by this brief

Read first:

- `docs/diagnostics/chaser_analytics_roadmap_2026-08-10.md`, especially
  "Chaser stimulus schedule", "Implementation status", and "Resolved
  semantics";
- `src/fisheye/docs/zarr_structure.md`, especially the stimulus/event layout;
- the existing Citrus-H5 stimulus/event ingestion and clock-alignment paths.

When the fixture is available, implementation should start from current `sun`
in a clean isolated worktree. Follow the repository `AGENTS.md`, use
`scripts/py` for every Python command, and keep each commit reviewable. Do not
push, merge, alter production selectors, or modify canonical archives without a
separate reviewed authorization.

## Why this exists

Citrus has reported a schedule mode named `fixed_n_min_gap_v1`: exactly N
randomly timed chases with a minimum onset gap. The historical behavior is
reported as `bernoulli_tonic_v1`; an absent mode field denotes that legacy
behavior. Both regimes must remain distinguishable conditions.

Palette does not yet import the reported schedule metadata or the complete
realized chase-event surface. The planned threat-aligned frame table and later
hazard/latency analyses require a versioned importer. The H5 artifact is the
source of truth; discrepancies between it and this brief must be reported and
resolved explicitly.

The proposed production candidate is not approved by this brief: training 300
s, `n_chases=4`, `min_gap=49 s`, `lead_in=45 s`, and `tail_guard=29 s`.
Importer code must read parameters and observed durations from source metadata,
never from these proposed values.

## Reported Citrus surface to verify

Planned schedule, per protocol step:

- `/chaser_schedule/planned/step_N`;
- schedule mode and parameters;
- seed and RNG algorithm (reported as `splitmix64_u53_v1`);
- episode bounds and cooldown metadata;
- planned onset times in training and step clocks.

Realized behavior is reported under `/events`, with `/trials/trial_index`
linkage. Verify the exact event names and fields in the H5. Reported events
include:

- scheduled onset due and onset missed;
- positioning start/end;
- chase-motion start/end;
- trial/onset linkage, observed step/training times, reason, duration, and
  `clock_source = protocol_step_steady_clock`.

Reported semantics metadata to verify:

- `planned_onset_semantics = "positioning_start"`;
- `chase_motion_onset_event = "CHASER_CHASE_MOTION_START"`;
- `positioning_visibility = "visible"`;
- `positioning_cue_kind = "movement_policy_change"` may be absent from the
  first fixture and must be recorded as found rather than invented.

## Required semantics

- Threat onset for response analyses is realized
  `CHASER_CHASE_MOTION_START`.
- Positioning is a distinct visible interval, not quiet time and not chase
  motion.
- Planned onsets are scheduler bookkeeping and must never become observed
  stimulus times.
- The importer must preserve absent, `bernoulli_tonic_v1`, and
  `fixed_n_min_gap_v1` modes without changing historical interpretation.

## Implementation goals

1. Define a closed, versioned logical contract, tentatively
   `palette.chaser_schedule.v1`, using the maintained exact array-contract
   representation. Freeze paths, dtypes, axes, units, null/fill semantics,
   parameters, clock source, and semantic attributes.
2. Preserve planned onsets and realized episode intervals separately. Store
   source-clock times and derived acquisition-frame indices with explicit
   domains and units.
3. Convert times with the existing clock-alignment authority. No hardcoded
   camera rate, training duration, or protocol-step duration is permitted.
4. Add a typed reader that makes realized chase-motion onset the obvious
   analysis default while keeping planned schedule access explicit.
5. Record planned-versus-realized QC: due/missed outcomes, reasons, realized
   count, and observed gap checks. Missed episodes are QC data, not structural
   corruption; malformed schedule/event linkage fails closed.
6. Publish through the standard hidden-stage, validate, atomic-rename,
   manifest-last lifecycle with exact provenance and completion eligibility.
   Decide explicitly whether this is part of the existing stimulus-import
   registry stage or requires a new maintained stage.
7. Add a contract document with an honest `contract-meta` implementation
   status at every checkpoint.

## Fixtures

The reported hardware-free fixture currently exists only on the Citrus rig:

`/tmp/citrus_chaser_schedule_example/headless_protocol_contract_smoke.h5`

It reportedly contains two planned onsets near 9.43724 s and 16.5384 s. Obtain
and hash this artifact before implementation. Its parameters are a smoke
fixture, not the proposed production protocol.

Also create minimal synthetic H5 fixtures for:

- absent legacy mode;
- explicit Bernoulli mode;
- fixed-N mode;
- fixed-N with a missed onset;
- malformed schedule/event linkage.

## Acceptance checklist

- [ ] The real fixture is available, hashed, and its observed schema is
      documented before production code is written.
- [ ] Fixed-N import produces the exact contracted Zarr surface.
- [ ] The typed reader returns realized chase-motion episodes whose source
      times exactly match the fixture.
- [ ] Frame indices come from existing clock-alignment evidence and round-trip
      within one source frame.
- [ ] Legacy and explicit Bernoulli fixtures retain their historical meaning.
- [ ] Missed onset produces QC evidence; malformed linkage fails with a
      specific error.
- [ ] No schedule parameter, camera rate, or step duration is hardcoded.
- [ ] The written run passes exact schema, provenance, atomic-publication,
      completion, and consolidated/direct metadata-equivalence validation.
- [ ] Focused tests and the relevant CI shards pass.

## Explicitly out of scope

- Threat-aligned frame-table, hazard, anticipation, or other scientific
  analysis implementation.
- Historical Bernoulli parameter backfill.
- Production protocol approval.
- Production selector or registry-authority activation.
