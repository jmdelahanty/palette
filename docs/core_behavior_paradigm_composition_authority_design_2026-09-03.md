# Core behavior and paradigm export authority design — 2026-09-03

<!-- decision-meta
status: accepted-design-review-synthesized-current-implementation-partial
created: 2026-09-03
last_updated: 2026-09-05
baseline_commit: 07db267c
review_checkpoint_commit: afbc1d0d6af822ca7cc4e3b051cdd9bc981df80c
composite_checkpoint_commit: 2ae7701e
review_method: six parallel read-only Luna xhigh audits plus primary-agent synthesis
scope: singular core-behavior authority selection, normalized validated-behavior
  exports, and chaser/moving-grating/other paradigm extensions
related: docs/validated_behavior_cohort_export_implementation_design_2026-08-31.md,
  docs/validated_behavior_chaser_appearance_export_successor_2026-09-01.md,
  docs/validated_recording_behavior_composition_design_2026-08-31.md,
  docs/cohort_release_workflow.md
-->

## Question this design answers

Palette already has two related but distinct export capabilities:

1. The published GoodBatBadBat Phase-C export combines ordinary motion, bout,
   and body-frame facts with chaser-specific facts. Its immutable generation
   contains 30 tables across 84 recordings. It is a real combined behavior and
   chaser dataset, not a prototype.
2. The in-progress complete core-workflow profile projects five sibling
   scientific grains from canonical recording-local publications:
   `kinematics_samples`, `subject_body_frame_samples`, `eye_trace_samples`,
   `tail_trace_samples`, and `canonical_swim_bouts`.

The remaining design question is not whether Palette can put ordinary behavior
and paradigm-specific analysis in one export. It can and has. The question is:

> How should complete core behavior and any paradigm-specific extension share
> one exported dataset without selecting competing motion, body-frame, or bout
> authorities or duplicating ambiguous facts?

## Decision

Every immutable exported dataset selects **exactly one authority per core
semantic capability** for each recording-scoped analysis unit. Paradigm-specific
analytics consume those selected authorities. They do not select, recompute, or
silently substitute a second authority for the same core capability.

The core capability roster is:

- acquisition frame and timing authority;
- subject position/track authority;
- track-kinematics authority;
- subject-shape and anatomical body-frame authority;
- eye-geometry authority;
- tail-kinematics authority; and
- canonical swim-bout authority, bound to the selected track-kinematics and
  frame-axis authority.

The roster is singular **per capability**, not one monolithic authority record
for heterogeneous scientific facts. Multiple candidate publications may exist
in a recording Zarr for review, comparison, recovery, or reprocessing. The
dataset manifest admits exactly one candidate for each required capability and
seals its concrete path and identity digests.

```text
acquisition frame/time authority
              |
              +--> selected subject track --> kinematics --> swim bouts
              |
              +--> selected subject shape --> body frame
                                           --> eye geometry
                                           --> tail kinematics

selected core roster + chaser authority ------> chaser-relative facts
selected core roster + grating authority -----> stimulus-aligned facts
selected core roster + feeding authority -----> feeding-relative facts
```

The existing `validated_behavior/v1` normalized-Parquet publisher, immutable
generation manifest, validation receipt, selector policy, and lazy reader remain
the only cohort export surface. Core and paradigm composition is expressed by
installed profiles and source admission. It does not create another publisher,
layout, selector, manifest family, or scientific Zarr authority.

## Terms

### Candidate publication

A completed immutable analysis run that may be eligible for selection. Several
candidates may coexist. Completion or a plausible name does not make a
candidate the authority selected by an export.

### Selected authority

The one candidate admitted for a semantic capability in one dataset generation.
Its run identity, manifest/receipt identity, row identity, coordinate authority,
and temporal authority are sealed into the recording bundle and cohort manifest.

### Projection

A deterministic, receipt-bound representation of selected authority data at a
declared row grain. A projection is not a new authority. Two projections may
legitimately expose different columns or grains while binding the same selected
authority.

### Paradigm extension

Tables whose scientific meaning additionally depends on a chaser, moving
grating, feeding, or other experimental authority. An extension binds the core
authority roster plus its paradigm authority and publishes only the added
relations, events, annotations, and summaries required by that paradigm.

## Current state and the source of apparent duplication

The GoodBatBadBat Phase-C profile already contains core-like projections:

- `provider_motion_samples` contains position, heading, speed, acceleration,
  and provider/chaser context;
- `body_frame_samples` contains acquisition-frame anatomical body geometry;
- `bout_detector_signal_samples` and `canonical_swim_bouts` expose the selected
  bout detector signal and events; and
- the remaining tables add chaser identity, occurrence, relative geometry,
  protocol, trial, epoch, association, occupancy, escape/freeze, and near-field
  facts.

The complete five-grain core-workflow profile uses different source-native
projections:

| Core capability | Complete core-workflow projection | Phase-C projection | Relationship |
|---|---|---|---|
| Track motion | `kinematics_samples` | `provider_motion_samples` | Related motion facts; different row/projection contracts and Phase-C provider/chaser context |
| Anatomical frame | `subject_body_frame_samples` | `body_frame_samples` | Related geometry; different source-row and timing semantics |
| Swim bouts | `canonical_swim_bouts` | `canonical_swim_bouts` | Same table contract name; source authority must be identical or one must be explicitly selected |
| Eyes | `eye_trace_samples` | no complete Phase-C counterpart | Additive |
| Tail | `tail_trace_samples` | no complete Phase-C counterpart | Additive |

Physical repetition across two independently published query datasets is
possible when both intentionally export the same recording. It is derived-data
duplication, not duplicate Zarr authority, but it still has storage and
interpretation costs. A future combined profile must not be constructed by
blindly unioning the Phase-C and five-grain table rosters.

### Reviewed Phase-C table classification

The 2026-09-03 parallel audit classified all 30 Phase-C tables at checkpoint
`afbc1d0d`. Phase C inherits Phase B and replaces only `chaser_occurrences`
with schema v2
(`src/fisheye/analytics_exports/validated_behavior_phase_c_contracts.py:1-7,79-85`).
The migration classification is:

- generic export envelope: `cohort_recordings`, `recording_bundles`, and
  `recording_capabilities`;
- overlapping core-like facts requiring subtraction or rebinding:
  `provider_motion_samples`, `body_frame_samples`, and
  `canonical_swim_bouts`;
- support/evidence projections: `recording_source_bindings`,
  `bout_detector_signal_samples`, and `controller_trial_gap_evidence`; and
- paradigm facts: `position_providers`, `chaser_occurrences`,
  `semantic_epochs`, `controller_trials`, `bout_chaser_associations`,
  `bout_response_distance_bins`, `trial_escape_freeze_events`,
  `trial_escape_freeze_summaries`, `trial_escape_freeze_threshold_sweeps`,
  `epoch_behavior_summary`, `spatial_occupancy_support`,
  `spatial_occupancy_bins`, `radial_near_field_summary`,
  `same_quadrant_occupancy`, `radial_near_field_density_bins`,
  `radial_near_field_distance_cdf`, `body_alignment_distance_bins`,
  `stimulus_native_state_support`, `chaser_relative_samples`,
  `body_relative_samples`, and `controller_trial_membership`.

This classification is a migration decision, not permission to copy the last
group unchanged. Every retained extension table still owes an exact dependency
on the selected core roster and corrected foreign keys/cardinality. In
particular, the present `body_relative_samples` contract targets the old
`body_frame_samples` grain
(`src/fisheye/analytics_exports/validated_behavior_phase_b_contracts.py:493-511`).

`recording_source_bindings` should be derived from the one composite bundle or
offered as a metadata view rather than copied from two profiles.
`bout_detector_signal_samples` may remain an optional receipt-bound diagnostic
projection; it is explicitly a detector response, not a physical speed
authority
(`src/fisheye/analytics_exports/validated_behavior_phase_b_contracts.py:444-453`).
`controller_trial_gap_evidence` remains valid paradigm evidence when it is
bound to the selected trial authority.

### What the overlap audit proved

- `provider_motion_samples` is keyed by provider track rows and carries
  provider roles, pixel and physical coordinates, multiple speed products,
  acceleration, angular motion, and provider/body/tracking row references
  (`src/fisheye/analytics_exports/validated_behavior_phase_b_contracts.py:78-149`).
  Its maintained Phase-C authority is provider-scope track kinematics. Core
  `kinematics_samples` is a source-rate standalone projection of an offline
  track-kinematics publication; every source frame is retained by default
  (`src/fisheye/analytics_exports/contracts.py:339-397` and
  `src/fisheye/analysis_workflows/validated_behavior_source_admission.py:86-92`).
  The two are related projections, not aliases. A future composite publishes
  core `kinematics_samples` once and regenerates any required chaser relation
  against that selected authority.

- `body_frame_samples` is one chaser/provider-oriented row per deduplicated
  acquisition frame with session-timestamp semantics
  (`src/fisheye/analytics_exports/validated_behavior_phase_b_contracts.py:303-336`).
  `subject_body_frame_samples` is one canonical subject-shape observation with
  subject-shape publication, row, temporal, camera, and coordinate authority
  (`src/fisheye/analytics_exports/validated_behavior_core_behavior_contracts.py:152-195`).
  They cannot be renamed into one another; chaser body-relative facts must be
  reprojected onto the selected core body-frame grain.
- Both profiles use the same Arrow contract for `canonical_swim_bouts`, but
  that establishes schema compatibility only. Phase C currently binds
  selector-ineligible provider-scope motion and a generalized-successor
  projection; core binds selector-eligible offline motion and the report-named
  bout event array directly. Their provenance fields and `bout_row_id`
  semantics also differ
  (`src/fisheye/analytics_exports/validated_behavior_adapters.py:1040-1105` and
  `src/fisheye/analytics_exports/validated_behavior_core_behavior_adapters.py:396-457`).
  They are **not presently proven to be the same bout authority**.

### Full-rate core-motion decision

The first sealed Sleepyfish five-grain generation used the earlier explicit
10 Hz workflow policy. That immutable generation remains valid as a sampled
query product, but it is not the full-rate colleague handoff. The maintained
core workflow now defaults to all source frames and records the bound source
rate with `sampling_stride_frames=1`. A lower-rate projection remains
available only when a caller explicitly supplies a sampling rate. This changes
neither the recording-local track-kinematics authority nor the shared
`validated_behavior/v1` publication surface.

### Core-motion ownership implementation checkpoint (2026-09-04)

The maintained five-grain core profile now owns the complete reusable physical
motion projection. Its `kinematics_samples` v2 contract adds the persisted
filtered and smoothed speed/path increments, signed tangential acceleration,
smoothed signed tangential acceleration, transition deltas, and cumulative
smoothed path distance. These values are copied from the selected
track-kinematics authority; the export does not recompute them.

The successor is deliberately a profile version, not a new publication
surface:

- new bundle sets default to `validated_core_behavior_five_grain_v2` on the
  existing `validated_behavior/v1` planner, sharder, atomic publisher,
  manifest, selector, validator, and reader;
- immutable v1 bundle sets continue to resolve through
  `validated_core_behavior_five_grain_v1`; its kinematics Arrow-contract digest
  and complete table-spec digest are frozen by regression tests;
- v2 source admission binds the exact persisted derivative and integral arrays
  and their physical/temporal authority records, then verifies their sealed
  array digests while those arrays are already resident for Parquet projection;
- the acceleration field is named
  `signed_tangential_acceleration_mm_s2` because it is the signed first
  difference of scalar smoothed speed, not a vector-acceleration magnitude;
- cumulative distance is explicitly the per-track cumulative sum of smoothed
  frame-path increments, with invalid transitions contributing zero; and
- installed export profiles fail closed if they contain both
  `kinematics_samples` and `provider_motion_samples`. A paradigm extension must
  join the selected core motion authority instead of installing a competing
  motion projection; and
- a real v2 profile writer-to-atomic-publisher-to-unpatched-lazy-reader
  regression seals the installed profile boundary and verifies the new motion
  fields without injecting table specs into the reader.

This checkpoint does not reinterpret or mutate existing Phase-C exports. The
chaser migration remains Track C/D work: subtract the overlapping provider
motion projection, rebind the retained paradigm relations to core row identity,
and prove their foreign-key cardinality before publishing a composite profile.

### Core-motion all-camera admission checkpoint (2026-09-04)

At `2026-09-04T20:55:00Z`, the new v2 core source resolver was run read-only
against the four completed Sleepyfish execution reports below, using published
consolidated metadata. The four checks ran concurrently on
`delahantyj-ws1.hhmi.org`; they did not create a plan, write scratch data,
mutate a Zarr, or change a selector.

```text
/groups/johnson/johnsonlab/jeremy/operations/
  sleepyfish_validated_core_behavior_full_rate_20260904_v002/source_reports/
```

All four recordings passed. Each resolution selected export profile
`validated_core_behavior_five_grain_v2`, motion surface
`core_motion_physical_v2`, all 27 required persisted motion arrays, and
projection-contract schema v3:

| Recording | Admission time (s) |
|---|---:|
| `2026_08_06_19_13_35_cam2010093` | 114.282 |
| `2026_08_06_19_13_35_cam2010094` | 113.993 |
| `2026_08_06_19_13_35_cam2010095` | 114.516 |
| `2026_08_06_19_13_35_cam2010096` | 114.053 |

This is source-admission evidence, not publication evidence. It proves that
the selected immutable track-kinematics publications already contain and bind
the physical motion surfaces required by the v2 projection, so no upstream
motion recomputation or scientific-Zarr migration is required.

### First full publisher canary and frozen-contract correction (2026-09-04)

The first selector-ineligible full publisher canary ran from CI-tested commit
`9e3fdcfdf113f799759c3ecdcf11cf7ebd383a8f` in its immutable deployment
worktree. The operation, bundle, and plan remain at:

```text
/groups/johnson/johnsonlab/jeremy/operations/
  sleepyfish_validated_core_behavior_full_rate_core_motion_v2_20260904_v004/
```

The bundle record digest is
`8049e4226d1189d49a324bc3191bb2cc4bcd37eec3fbcf015c02be6849228fa5`,
the capability-matrix digest is
`5e1e582229250ffb966d772328351eb9485f59c4b075d20266a4c513c632fbd5`,
and the plan digest is
`033a28a41f0cfe64a03c8507466bffef050cb1b5066b99436c123e725f6710a6`.
All safety fields are false.

LSF shard array `154008697` ran all four members concurrently; each failed
closed after 164--200 seconds with the same error before any shard receipt or
publication was created. Dependent finalizer `154008698` therefore did not
publish:

```text
ValueError: Core-motion projection differs from the installed successor contract.
```

The scientific records were identical. Production bundle validation had
recursively frozen JSON objects as read-only mappings and JSON arrays as
tuples, while the adapter performed a shallow comparison against ordinary
dictionaries and lists. Commit `d7cdb1da` corrects that representation
boundary by recursively thawing only JSON containers before exact comparison;
it does not coerce scalar values. A production-shaped regression proves that
the frozen valid contract is accepted and that a nested semantic change still
fails closed. The two focused export suites pass 44/44 locally. This correction
still requires complete CI and a fresh immutable v005 deployment, bundle,
plan, and canary; the failed v004 evidence will not be edited or reused.

### Successful replacement full publisher canary (2026-09-04)

All required CI gates and all 16 non-GPU shards passed on runtime commit
`4077579afc8cd29922b7d8cbe22f6178a9b7a154`. The deployment helper then
created and locked the detached worktree below, verified its import root, and
left the shared `/groups` checkout unchanged:

```text
/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/
  core-motion-authority-20260904-4077579a
```

The fresh v005 operation is:

```text
/groups/johnson/johnsonlab/jeremy/operations/
  sleepyfish_validated_core_behavior_full_rate_core_motion_v2_20260904_v005/
```

Its bundle record digest is
`3e8111a6b29c8d432331be0815498b0985f0af330c8ea77ac314fdcea4ba44aa`.
Its capability-matrix digest remains exactly
`5e1e582229250ffb966d772328351eb9485f59c4b075d20266a4c513c632fbd5`,
proving that v004 and v005 selected the same scientific authorities. The new
plan digest is
`3ca8af2079892b255b0677b22f15e743c8ab38232da7688b5a092540cf250c8d`.

LSF shard array `154008759` ran all four recordings concurrently on compute
hosts and completed in 432--468 seconds per recording. Every stderr file is
empty, and all four schema-v2 shard receipts passed semantic validation.
Dependent finalizer `154008760` then completed in 48 seconds. Its internal
telemetry records 5,882,107,589 bytes across 32 parts, 37.283 seconds for the
necessary destination copy-and-hash pass, 0.097 seconds for receipt
composition, 0.907 seconds for receipt-only precommit plus atomic commit, and
38.779 seconds total.

The manifest-last publication is `complete_selector_ineligible`, retains all
six false safety fields, and has record digest
`9e06161f8aa0f63f7154fc6e22f75191a6af00cbe13ad57fb593c662bb48c386`.
Its transfer-receipt digest is
`3e632a616b93f5c261cd9f102bf75ece3f79c0afa066a291187ddd73d3644a67`,
and its validation-receipt digest is
`7fa0a0fec0e85d3d668a939baaf4aed7228f7d44e619fcbf1a4819023cb1680a`.
The generic unpatched reader reopened the publication in receipt mode and
performed a bounded projection of the new speed, signed-acceleration, and
cumulative-distance columns.

The sealed row counts are:

| Table | Rows |
|---|---:|
| `kinematics_samples` | 11,469,925 |
| `subject_body_frame_samples` | 11,469,925 |
| `eye_trace_samples` | 11,750,416 |
| `tail_trace_samples` | 114,699,250 |
| `canonical_swim_bouts` | 79,235 |
| `recording_capabilities` | 24 |
| `cohort_recordings` | 4 |
| `recording_bundles` | 4 |

This closes the core v2 writer-to-publisher-to-reader canary. It does not
activate a selector, update a registry, mutate source Zarrs, or authorize the
future chaser-composite profile described in Track C.

### Core-authority roster implementation checkpoint (2026-09-05)

Implementation started from merged `main` commit
`68f7b7a7cff64f2483df2b39c135dcb369ca82ea`. The first tranche adds the shared
authority boundary required before any composite profile is installed:

- `build_core_authority_roster()` seals the exact six-capability map already
  produced by `bind_core_behavior_cohort_sources()`, including the execution
  receipt and cross-grain join authority. It does not rediscover a selector or
  introduce another publisher;
- the bound core source now exposes that roster and one normalized
  `BoutAuthorityIdentity` constructed from the strict selected bout source;
- `bind_core_motion_and_bouts_from_roster()` reopens only the exact motion and
  bout dependency closure through the same strict binders. It rejects a stale
  roster/source mismatch and does not open unrelated eye or tail sources;
- `CoreMotionTrackSourceHandle` exposes only the roster-selected track and its
  declared motion arrays. It requires an explicit track ID and a sealed
  consumption receipt; selector names, fallback paths, and implicit track zero
  are not representable at this boundary;
- a sealed downstream consumption receipt names the roster digest, required
  capabilities, and exact selected track. The cross-grain join authority is
  mandatory, so a paradigm producer cannot bind motion by frame number alone;
- the bout comparator returns `equal`, `conflict`, or `not_proven`. Equality
  includes publication, motion, row bounds, FPS/frame axis, candidate/signal,
  selected-event dtype/count/content, and binding identity; identical event
  bytes do not erase a conflicting motion authority; and
- `compose_disjoint_table_specs()` rejects a table collision before inserting
  either component into a combined map, closing the dictionary-overwrite hole.

The focused authority/core-export/chaser-planner suites pass 39/39 on the
workstation. A
read-only real-source resolution of Sleepyfish camera 2010093 through the
ordinary unpatched core resolver completed in 35.312 seconds and produced core
roster digest
`2046ca97d8439483e4f752f00f27b5deef4bea21c0980306297206989d036dbc`
and bout-identity digest
`430b2e6c19e84589a99b718f414a0784b8299dd0b3658c5371ae5c0cf14031a8`.
Reopening only the receipt-selected motion/bout dependency closure from that
roster completed in 6.717 seconds and returned exact selected track `0`. Both
reads used consolidated metadata and performed no writes.

This checkpoint deliberately does **not** install the composite profile or
reinterpret existing Phase-C products. The current Phase-C chaser facts remain
provider-motion/body/bout-bound; they must be reprojected against the selected
core roster before admission. Until that migration is complete, composition
must fail closed rather than treating equal-looking motion columns or event
rows as shared authority.

### Core-bound chaser-relative foundation checkpoint (2026-09-05)

The first chaser re-projection boundary now feeds the existing
`chaser_relative_frame` computation, prepared-candidate schema, atomic Zarr
materializer, and strict source handle. It does not define a new run family,
publisher, selector, or export table. The existing manifest context gained one
optional digest-enveloped `core_authority` record; old publications remain
readable unchanged, while a core-bound candidate seals the selected roster,
consumption receipt, exact track, source chaser publication, and row join.

This boundary deliberately consumes an existing chaser-relative publication
only for chaser positions, identities, occurrence, trials, timestamps, and
controller evidence. Its historical fish-position and body-frame authorities
are stamped `not_used_core_roster_selected_instead`. Fish positions come from
the roster-selected core track and every chaser frame must resolve exactly one
row of that track; missing frames, coordinate conflicts, timing conflicts, or
physical-scale conflicts fail before preparation. The relative result contains
no speed, acceleration, cumulative-distance, or other repeated core-motion
fact. Its pixel-space fish position is a bounded re-expression of selected
`positions_mm` under the exact shared physical-scale authority.

The second tranche binds the roster-selected `subject_body_frame_samples`
capability through the ordinary strict subject-shape loader and mints a
receipt-bound process-local body handle. Chaser rows join to body observations
only by the declared `(recording_id, source_acquisition_frame_index,
source_instance_key)` key. A missing observation remains explicit `-1`/NaN
evidence, duplicate compound keys fail closed, and interpolation, neighboring
body frames, motion-heading substitution, and the historical chaser body
authority are all prohibited. Motion and body handles must carry the identical
consumer/capability/track receipt from one roster. The existing body extension
and publisher carry the result; no second body schema or publication surface
was introduced.

The maintained planner still uses `MOTION_BOUT_PAIRS` at this checkpoint and
therefore remains explicitly incomplete. It must switch atomically to the
roster-selected motion, body, bout, and explicit track identities; the old body
authority is not a fallback during that migration.

The third tranche removes the need to publish that historical intermediate in
new core-bound work. The established proxy coordinate/controller extractor can
now hand its validated chaser-only facts directly, in memory, to the core-bound
adapter. Its transient fish/body calculations are ignored and are never
published; the final payload substitutes only roster-selected motion and body
before entering the same existing atomic `chaser_relative_frame` publisher.
The existing materializer CLI exposes this as an explicit all-or-none core
mode requiring an exact roster file, expected roster digest, and explicit track
ID. Core-mode failure propagates and cannot fall back to proxy fish/body
authority; supplying a legacy body-frame run in core mode is rejected before
source preparation.

Static admission also has one shared conversion from a generic, already
validated full-rate core bundle member to its sealed authority roster. The
conversion checks the installed profile and capability contract, complete
six-capability closure, exact execution-report receipt, and binding-inventory
digest. It reuses the roster builder rather than giving the maintained planner
another authority grammar. Dynamic execution must still reopen that roster
through the shared resolver before any dependent scratch write.

Focused workstation validation passed 65/65 core-roster, chaser storage/source,
proxy-adapter, and core-mode CLI tests. A real
prepared-writer-to-atomic-materializer-to-unpatched-source-handle regression
also proves that the optional roster context survives the existing publication
surface with its digest intact.

### Maintained chaser planner migration checkpoint (2026-09-05)

The maintained GoodBatBadBat cohort planner now freezes task schema v8 from one
exact generic core-bundle set. Static planning selects the exact bundle member,
derives its roster through the shared core-bundle adapter, derives the sole
motion/bout track from that roster, and seals a consumer receipt. It no longer
contains `MOTION_BOUT_PAIRS`, `matches[0]` pair precedence, a provider-derived
body-frame choice, or implicit track zero. Historical task schemas remain
readable but cannot execute or mint a canonical successor.

Execution reopens the exact bundle-set and membership files and reconstructs
the frozen roster binding before creating scratch or receipt directories. The
epoch summary, relative-frame materializer, generalized bout response, and
escape/freeze chain receive the roster path, expected digest, and selected
track as an all-or-none input. Core-mode failure cannot fall back to the legacy
provider-motion or independent swim-bout inputs. A shared motion dependency
record validates the complete capability-digest roster and seals the selected
motion and bout publications into each motion-dependent scientific manifest.

`scripts/check_paradigm_core_authority_access.py` is wired into CI and rejects
reintroduction of the retired pair resolver, provider-motion/bout CLI
selection, or literal track zero in this maintained planner. It also requires
the static and dynamic shared resolver calls and the three frozen core CLI
arguments. Focused workstation validation passes 85/85 tests across the core
roster, proxy/relative adapter, maintained planner, composable operator, epoch
summary, generalized bout response, escape/freeze, handle adapters, and the new
ratchet.

This checkpoint is still intentionally incomplete. Before the branch is
merge-ready, every transitive chaser publication and reusable-output gate must
prove the same roster digest explicitly, the composite export profile and
foreign-key contracts remain Track-C work, and required CI has not yet run.

### Transitive chaser authority and reuse checkpoint (2026-09-05)

The maintained chaser dependency closure now carries one compact sealed
`palette.core_behavior.paradigm_relative_frame_dependency` record derived from
the already-validated `ChaserRelativeFrameSourceHandle`. This is a projection
of the shared resolver result, not another authority selector or artifact
grammar. The projection verifies the full core binding, exact consumption
receipt, recording/archive identity, selected track, motion and body source
bindings, chaser source, pixel-space conversion, and analysis profile before
sealing the smaller downstream record.

Controller trials, radial/near-field products, body alignment, gaze tracking,
generalized bout response, and escape/freeze now preserve that exact dependency.
Spatial occupancy requires its two providers to be either both legacy or both
core-bound; two core-bound providers must name the same roster, receipt, track,
motion binding, and body binding. Mixed core/legacy motion is rejected, and a
dependent producer cannot select or substitute another core source.

Reusable-output discovery is now authority-aware. For each core-bound stage,
the executor requires a digest-validated publication manifest containing the
exact frozen `core_authority_roster_sha256`; an absent, malformed, or different
claim blocks reuse. Near-field visits additionally prove exact relative-frame
and radial-child bindings. This is a cheap immutable-identity gate, not a
replacement for dynamic resolver admission.

Planning can no longer mark a task complete from existing paths and plot
receipts. Even when every expected output exists, the task remains runnable as
`validation_only`, causing `run-one` to reopen the frozen bundle selection and
validate each reusable output before any scratch write. Thus path presence is
only a reuse candidate, never authority evidence.

The CI ratchet now checks both sides of this boundary: the planner must invoke
the shared source validator and pass the exact expected roster into reusable
output checks, while each maintained transitive successor must project and seal
the shared dependency. Focused workstation validation passes 160/160 tests
across the roster/resolver, planner, ratchet, direct successors, and downstream
analytics. At that checkpoint, required repository CI had not run and the
composite export profile and cross-table foreign-key contract remained Track-C
work; the next checkpoint records their implementation.

### Core-plus-chaser composite export checkpoint (2026-09-05)

Commit `2ae7701e` installs the first composite profile without adding a cohort
publisher, selector, manifest family, dataset layout, or reader. The profile is
`validated_core_behavior_chaser_v1` on the existing
`validated_behavior/v1` surface. It contains 30 tables and 27 scientific row
projectors: the complete five-grain core suite plus the collision-checked
chaser-only subtraction of Phase C.

The per-recording composition envelope requires exactly two typed admissions:
one complete core-workflow execution report and one exact-chaser projection
receipt. It resolves both through their existing shared binders, requires
full-rate core motion (`sampling_stride_frames=1`), and proves that every
maintained chaser child names the selected core roster, track, motion, body, and
bout identities. The old Phase-C-compatible bundle continues to use the same
shared exact-chaser receipt resolver; no second child-receipt grammar was
created. The existing per-recording bundle CLI and generic cohort CLI dispatch
from the typed admission roles rather than adding a composite publisher CLI.

Composition omits `provider_motion_samples`, `body_frame_samples`, the Phase-C
`canonical_swim_bouts`, and their support projections. Core
`kinematics_samples`, `subject_body_frame_samples`, and
`canonical_swim_bouts` are emitted once. Chaser-relative rows bind the selected
core motion row with a declared foreign key. Body-relative rows preserve the
existing explicit `body_source_row_id=-1` failure evidence while projecting
only valid observations into a nullable foreign key to
`subject_body_frame_samples`; null keys are unconstrained using ordinary
relational foreign-key semantics, while every present key must close exactly.

The normalized bout comparator remains available for explicit migration or
comparison work, but the maintained composite does not admit a second legacy
bout table that would need deduplication. Its generalized bout, escape/freeze,
and epoch descendants instead retain the selected core motion-and-bout
dependency directly. This is stronger than claiming equality between two bout
authorities from similar-looking rows.

Focused workstation validation passes 93/93 composite, core, Phase-B,
Phase-C, bundle, generic-writer, atomic-publisher, and lazy-reader tests. Both
the core-only and composite profiles pass a real generic shard writer ->
receipt-composed publisher -> unpatched installed-profile reader round trip.
Import boundaries, the paradigm-authority ratchet, the tail-receipt ratchet,
the Zarr-open-mode ratchet, and observed-metadata-literal checks also pass.
Required repository CI and a real selector-ineligible composite canary remain
pending; no production publication or selector changed at this checkpoint.

## Composition contract

### 1. One reusable core authority roster

Source admission resolves a recording's core authorities once. The resolved
record is immutable and contains explicit `used`, `not_used`, `unavailable`, or
`blocked` state for every declared capability. Unknown or conflicting evidence
blocks only the requested workload's dependency closure.

Static planning proves that the selected producer profiles can issue the
required receipts. Dynamic admission verifies the concrete paths, receipts,
digests, selector state, and consolidated metadata generation after publication.
No predicted digest is accepted as authority evidence.

### 2. One base set of core fact tables

A combined future export writes the complete core fact tables once. These are
the reusable inputs for protocol-independent queries and downstream paradigm
tables. The scientific grains remain separate; shared frame keys do not by
themselves authorize joins.

### 3. Paradigm-specific tables are extensions

Chaser, moving-grating, feeding, and later paradigms contribute only their own
authorities and derived relations. Each extension declares:

- the exact core capabilities it consumes;
- the paradigm authority it adds;
- expected join keys and cardinality;
- units, coordinate frame, timing semantics, and validity policy; and
- typed unavailable/failure behavior.

Paradigm tables reference the selected core rows through declared foreign keys
or digest-bound lineage. They do not fall back to a different core source when
the selected source is unavailable.

### 4. Repeated columns never imply a second authority

A paradigm table may carry a small number of core values for bounded query
convenience only when its projection receipt binds the selected core authority
and validation proves exact equality for the declared relation. Such columns
are documented denormalized projections. They cannot be used to select or
override authority.

Large repeated motion/body payloads should instead be removed from the future
composite extension or exposed as reproducible joined views over the normalized
facts. A convenient wide view is a query adapter, not another publication.

### 5. Bout selection fails closed

The archive may contain several swim-bout runs, candidates, detector-signal
variants, or generations. A recording bundle selects one bout authority bound
to the selected track, track-motion manifest, frame axis, candidate, signal,
and event table.

When two input profiles name `canonical_swim_bouts`, composition compares at
least:

- run path and immutable publication identity;
- track ID and track-motion manifest/verification identities;
- frame-axis identity;
- candidate and detector-signal identity; and
- event-table contract and content identity.

If all required identities match, the composite publishes
`canonical_swim_bouts` once. If they differ, planning returns a typed authority
conflict. It never concatenates the rows, chooses by path/name/recency, or calls
both sources canonical. Publishing intentionally different bout methods would
require separately named method-comparison tables and is outside this contract.

The comparison is implemented through one normalized `BoutAuthorityIdentity`,
not through equality of profile-specific provenance column names. It contains
the recording and analysis-Zarr identity; run path, schema, layout, lifecycle,
and immutable publication identity; motion scope, run, manifest, verification,
track, and row bounds; FPS and complete frame-axis identity; candidate and
signal identity; selected-event dtype, count, and content digest; and the
binding/receipt digest. For a Phase-C source, selected event content must be
reopened through its exact loader and digested before equivalence can be
claimed. Missing identity means `not_proven`, never equality.

The core and Phase-C contracts currently attach different meanings to fields
with the same names. For example, core `source_manifest_sha256` identifies the
swim-bout array manifest while Phase C uses the generalized-successor manifest;
core `bout_row_id` is the source-array index while Phase C uses the generalized
association ordinal. The comparator therefore works on normalized scientific
identity, not raw row dictionaries.

## Relationship to existing Phase C

The completed Phase-C generation remains a valid immutable export under the
contract and source bindings with which it was published. This decision does
not mutate, reinterpret, or withdraw it.

Phase C already validates strong relationships among provider motion, body
geometry, and canonical bouts. Its existence proves the generic publisher can
carry base behavior and paradigm-specific facts together. It does not prove
that its provider-oriented projections and the complete core-workflow
projections are interchangeable.

Future chaser composition should be installed as a profile made from:

1. the reusable complete core authority roster and base tables; plus
2. a chaser extension containing the paradigm-specific tables whose inputs bind
   that same roster.

Compatibility readers may continue to read Phase C. New work must not create a
second `Phase-C plus five-grain` publisher or rewrite the sealed Phase-C
generation in place.

## Relationship to the in-progress Sleepyfish export

The four-camera Sleepyfish delivery has no chaser or moving-grating authority.
Its required product is the complete five-grain core-workflow profile on the
existing `validated_behavior/v1` surface. It should not carry empty chaser
tables or manufacture a paradigm identity merely to reuse Phase-C naming.

The in-progress implementation at the baseline above adds:

- exact recording filtering and workload-local admission;
- a core-workflow execution-report resolver;
- a recording bundle binding the five selected source publications and their
  cross-grain join evidence;
- table contracts and bounded projections for the five grains; and
- dispatch through the existing cohort planner, shard writer, atomic publisher,
  manifest, validator, and reader.

This implementation is the intended reusable core base, but at document
creation time it is branch-local and not yet CI-qualified. It must not be
described as production-ready or used for selector activation until required CI
passes.

### Read-only real-source checkpoint

On 2026-09-03, a read-only consolidated-metadata admission attempt used the
camera-2010093 completed execution report and its immutable analysis Zarr. The
execution-report admission itself passed. The full five-source resolver was
stopped by the operator after 12 minutes without a contract error because it
was still recomputing an entire tail measurement-array content digest through:

```text
bind_core_behavior_cohort_sources
  -> bind_tail_trace_sources
  -> load_tail_kinematics_coordinate_publication
  -> load_bound_array_measurement_descriptor
  -> validate_array_measurement_descriptor
  -> array_measurement_payload
  -> array_payload_sha256
  -> node[:]
```

This is incomplete canary evidence, not successful real-source admission. It
also identifies a concrete performance question for Track A: determine whether
the immutable tail publication's sealed array/measurement receipts can satisfy
this read boundary without replaying the full decoded hash. Any optimization
must preserve exact authority validation and fail closed for absent, stale, or
unsupported receipts.

The read-only receipt audit answered that question for this canary: its tail
publication does **not** expose the complete supported tail payload receipt
pair. Receipt-bearing tail publications already enter
`_run_tail_array_digest_scope`, where identity, path, dtype, shape, and
canonicalization are checked before sealed per-array digests are reused without
decoded payload reads
(`src/fisheye/shared/tail_coordinate_publication.py:258-300,2331-2348` and
`src/fisheye/shared/coordinate_frame_record.py:739-767`). Missing receipts take
the exhaustive compatibility path.

That compatibility path is especially amplified: the audit counted 116 tail
array hash calls, and each no-evidence hash performs an initial read plus an
immediate mutation-detection reread. The observed source can therefore incur
roughly 232 complete decoded reads before unrelated direct reads are counted
(`src/fisheye/shared/coordinate_frame_record.py:672-721`). A manifest or
measurement-descriptor digest cannot be treated as a payload receipt; it lacks
the independent physical-copy and mutation-exclusion proof.

The production design is:

1. every maintained tail-kinematics producer issues the complete sealed payload
   receipt through the atomic materializer;
2. the maintained tail loader, activation path, and core cohort binder require
   that receipt profile unconditionally;
3. absent, partial, stale, wrong-run, or wrong-manifest receipts fail before
   decoded tail-array validation;
4. there is no receipt-free tail-kinematics loader; the independent posture
   artifact retains its own kind-derived validation contract, and a CI access
   ratchet rejects any caller-selectable receipt bypass; and
5. an existing immutable receipt-free publication is never retrofitted in
   place. The four Sleepyfish recordings need newly published receipt-bearing
   successors from the maintained materializer before core-cohort admission.

Receipts eliminate repeated proof-of-identity decoding, not scientific data
access itself. Tail computation, Parquet projection, an explicitly requested
deep audit, and small semantic-axis checks still read the arrays they actually
use. The prohibited pattern is replaying a complete decoded hash merely to
re-prove an immutable publication at each planning or loading boundary.

Implementation checkpoint on 2026-09-03: the public kinematics publisher now
requires the atomic materializer's scan, installed-path, and verified-copy
evidence; activation and the maintained loader require the sealed receipt pair;
the direct archive CLI is fail-closed; and
`scripts/check_tail_payload_receipt_access.py` prevents maintained code from
reintroducing receipt-free loading, making receipt policy caller-selectable, or
bypassing the atomic materializer. The four Sleepyfish receipt-bearing tail
successors were subsequently published from CI-tested commit
`289e9ddc1f445ee44c30b9f25fa6ede924c24d2d` as the camera-specific
`tail_kinematics_sleepyfish_2026_08_06_core_behavior_receipt_v011_289e9ddc_*`
runs; direct and consolidated selector metadata agree for all four.

A later real-source five-grain resolver canary exposed one additional
producer/consumer mismatch in those v011 successors.  Their sealed payload
receipt pair is valid, but the ordinary maintained tail writer had stamped the
exact `tail_kinematics_array_schema` declaration only for the explicit
byte-planned candidate profile.  The strict tail export therefore rejected
v011 before reading scientific arrays with `Tail source lacks its exact
array-schema manifest.`  A completed workflow report and selector eligibility
were correctly insufficient to override that missing evidence.

The ordinary and byte-planned writers now share the same logical schema
obligation: every newly created tail run stamps the profile-specific exact
array declaration, and the atomic materializer validates it before publication
regardless of physical storage profile.  Existing v011 artifacts remain
immutable and are not retrofitted; the four recordings require new ordinary
successors before five-grain cohort admission.  This is a metadata-contract
repair in the producer followed by normal immutable republication, not a
receipt bypass or a new tail-loading profile.

The camera-2010093 v012 successor canary then completed on
`h06u26.int.janelia.org` in 193.56 seconds, published its exact ordinary array
schema and payload receipt pair, passed registry finalization, and exposed a
second boundary defect without weakening admission.  The tail-trace binder was
still searching loose tail-run and archive-root FPS attributes even though the
sealed subject-shape temporal authority already binds the canonical
acquisition-camera record and its 30 Hz source-video metadata.  Consequently,
the real five-grain resolver rejected the otherwise valid v012 source with
`Tail trace export requires one positive finite source FPS.`

Maintained tail-trace binding now has one rate-authority path: subject-shape
publication → temporal authority → acquisition camera-frame record → sealed
`source_video_metadata.fps`.  It requires the acquisition record reference and
digest, requires the same recording identity, and compares that rate exactly
with the track-kinematics publication.  Loose tail/root FPS probing has been
removed rather than retained as fallback.  A regression supplies a conflicting
root FPS and proves it is ignored, then removes the canonical temporal
authority and proves binding fails closed.  This consumer repair makes the
existing immutable v012 artifact consumable as published; it does not mutate or
republish scientific data.

Subject-shape publications follow the same maintained-consumption rule. Both
supported payload profiles are receipt-bearing: v1 records the sealed result of
the post-binding decoded scan, while v2 composes staged-transfer and final
binding evidence. Profile coexistence does not authorize a receipt-free third
path. Maintained loading, activation, and completed-publication validation must
reject an absent or partial receipt pair before scientific array traversal.

## Resolver and adapter boundary

Authority admission and scientific projection are separate responsibilities,
but they must not become separate sources of truth:

- the source resolver validates each supported publication profile at full
  strength and returns the sealed core authority roster;
- the planner invokes that resolver in read-only/evidence mode and persists its
  result in the plan;
- the table projectors consume only the persisted selected roster and reopen
  the exact sources through the same strict binders; and
- the generic publisher consumes projected shards and receipts without knowing
  Sleepyfish, GoodBatBadBat, chaser, or grating semantics.

An adapter may translate a source profile into the common roster or project an
installed table contract. It is not allowed to monkey-patch loaders, bypass a
gate, select a source, invent lineage, or publish through a parallel path.

### Concrete composite shape

The generic export plan and manifest each bind one bundle set. The composite is
therefore one new installed profile plus one composite bundle adapter, not two
bundle sets joined after planning and not a merger of two completed manifests.
For every admitted member it requires:

- exactly one supported core execution receipt;
- zero or more explicitly supported paradigm-extension receipts;
- one normalized core capability map resolved through the existing
  `bind_core_behavior_cohort_sources()` path;
- one deterministic `core_authority_roster_sha256` over that normalized map;
- extension bindings that name that digest and their own paradigm authority;
  and
- no duplicate receipt roles, unsupported role combinations, or replacement
  core authorities supplied by an extension.

The central admission dispatcher remains the only receipt grammar boundary
(`src/fisheye/analysis_workflows/validated_behavior_source_admission.py:22-35,383-430`).
The existing generic bundle-set schema can seal the additional capability
bindings and inventory without a manifest-schema change
(`src/fisheye/analysis_workflows/validated_behavior_cohort.py:1116-1339`).
The frozen cohort query remains membership-only; authority selection happens
after freezing during source admission, not in registry query syntax
(`src/fisheye/cohorts/spec.py:259-330`).

The composite profile is assembled explicitly as generic envelope tables, the
five core tables, and audited paradigm-only tables. It must check table-name
collisions before constructing a mapping. A Python `{**base, **extension}`
merge can overwrite a duplicate before the existing table-spec validator sees
it, so validation after such a merge is not sufficient
(`src/fisheye/analytics_exports/validated_behavior_contracts.py:260-297`).

Old Phase-C bundle adapters retain their exact one-receipt contract. The
composite receives its own narrow adapter branch; legacy branches are not
relaxed. The selected profile ID continues to route through the closed profile
registry, the existing generic plan/shard/publish path, and the generic reader
(`src/fisheye/analytics_exports/validated_behavior_profiles.py:48-55,74-117`).
An older reader safely rejects an unknown composite profile; a current reader
must continue opening sealed Phase-C v1 publications unchanged.

### Maintained-path divergences to remove

The review found functioning scientific machinery behind several independent
selection surfaces. They are compatibility behavior today, but they are not
permitted in a maintained composite dependency closure:

- the composable chaser planner resolves hard-coded motion/bout pairs and takes
  `matches[0]`
  (`src/fisheye/utils/materialize_composable_chaser_successor_cohort.py:425-438`);
- stimulus response independently resolves the latest track-kinematics and
  swim-bout runs (`src/fisheye/analysis/stimulus_response.py:394-405,924-993`);
- older chaser distance, bout-response, epoch-summary, and escape paths retain
  explicit/latest fallback resolvers; and
- stimulus, movement-bout, and Megabouts paths contain implicit
  `track_id=0` defaults rather than consuming a selected track identity.

Migration replaces those decisions with the persisted core roster. Explicit
paths are acceptable only as inputs to the shared resolver and must match the
selected identity. `latest`, ordered-first, implicit track-zero, or fallback
selection after an authority mismatch is a typed planning failure. Static CI
ratchets should prevent their reintroduction in maintained composite planners.

The migration order is chaser first, then the shared stimulus-response path
(moving grating, concentric/radial OMR, and looming), then generic projections
such as activity/spatial bins and classifier-specific extensions. Paradigm
algorithms remain responsible for genuinely additional facts—stimulus state,
relative geometry, trial/epoch relations, OMR gain and latency, response
classification, occupancy, escape/freeze, and similar outputs—but not for
reselecting core motion/body/bout authority.

## Invariants

1. One dataset generation has at most one selected authority for each core
   capability and recording-scoped analysis unit.
2. Every downstream capability names the exact core authority-roster digest it
   consumed.
3. A paradigm extension cannot become admitted when a required core capability
   is unavailable or conflicts with the selected roster.
4. Table names do not establish authority identity; receipts and manifests do.
5. Shared frame keys do not establish join safety. Recording, camera, track,
   row, coordinate, temporal, and source identities plus cardinality must close.
6. `canonical_swim_bouts` occurs once in a composite table roster.
7. A completed source artifact is not admitted solely because it is complete,
   selector-visible, nearby, or plausibly named.
8. The Zarr publications remain scientific authorities. Parquet tables are
   immutable query/interchange projections.
9. Existing immutable generations are never changed in place.
10. All profiles use the same generic `validated_behavior/v1` publication and
    reader surface.
11. Table-name collisions are rejected before profile mappings are assembled;
    dictionary overwrite is never treated as composition.
12. Maintained composite planners never select core authority by `latest`,
    ordered-first match, implicit track zero, filename, or proximity.
13. A payload receipt can replace decoded payload hashing only for the exact
    receipt profile and immutable publication it proves; metadata self-digests
    are not promoted into physical evidence.

## Implementation plan

### Track A — finish the complete core base

- [x] Finish the five-grain source resolver and recording-bundle schema.
- [x] Require one exact authority for every required core capability.
- [x] Seal the cross-grain join record and bind every capability to it.
- [x] Reuse one strict source binding within a recording to avoid repeated
      whole-source validation.
- [x] Make maintained tail publication, activation, loading, and core binding
      receipt-only; fail before decoded reads when evidence is absent or stale.
- [x] Retire direct archive publication and reject receipt-free loaders,
      caller-selectable receipt policy, and low-level writer access with an
      independent CI ratchet.
- [x] Publish new receipt-bearing successors for the four receipt-free
      Sleepyfish tail publications through the maintained atomic materializer;
      the four v011 successors were published from `289e9ddc` without mutating
      the originals.
- [x] Keep the five scientific grains in separate normalized tables.
- [x] Route planning, sharding, publication, validation, and reading through
      the existing generic cohort engine.
- [x] Prove that no new publisher, selector, manifest family, or CLI path was
      introduced.
- [ ] Add a real execution-report-to-resolver boundary test.
- [x] Add a real generic-writer-to-publisher-to-unpatched-reader boundary test
      for the installed v2 profile.
- [x] Run a read-only admission canary for all four Sleepyfish cameras.
- [x] Run every required CI gate on runtime commit `4077579a` before its
      detached deployment and selector-ineligible canary.
- [x] Run and validate the real four-camera selector-ineligible v005
      writer-to-publisher-to-unpatched-reader canary.
- [ ] Run required CI on the final documentation head before merge.

### Track B — audit paradigm composition

- [x] Inventory the exact authority bindings and table grains in Phase C.
- [x] Classify every Phase-C table as core fact, paradigm extension,
      denormalized convenience projection, metadata, or summary.
- [x] Compare `provider_motion_samples` with `kinematics_samples` field by
      field and identify the reusable base versus chaser-only additions.
- [x] Compare `body_frame_samples` with `subject_body_frame_samples` including
      row, timing, and validity semantics.
- [x] Determine whether the current contracts prove Phase-C and core-workflow
      bout identity. Result: no; provider/offline scope and event-content
      evidence differ.
- [x] Specify the candidate chaser extension roster after core-table
      subtraction.
- [x] Inspect moving-grating and other maintained workflows for independent
      motion/body/bout selection that should instead consume the core roster.
- [x] Propose compatibility and versioning rules without rewriting Phase C.

Track B was completed as six parallel read-only reviews at checkpoint
`afbc1d0d`, covering table overlap, resolver architecture, bout identity,
maintained paradigm consumers, immutable compatibility, and receipt hot paths.
It changed no code, Git state, or publication data. Runtime equivalence for any
particular legacy/core bout pair remains an explicit Track-C proof, not an
inference from this static audit.

### Track C — implement a composite only after Track B

- [x] Register one explicit composite profile on `validated_behavior/v1`.
- [x] Add one composite bundle adapter requiring exactly one core receipt plus
      a closed set of supported extension receipts per admitted member.
- [x] Derive and seal `core_authority_roster_sha256` from the normalized core
      capability bindings.
- [x] Require every extension publication to name the selected roster digest.
- [x] Compose the complete core specs with paradigm-only extension specs using
      a collision-checking helper rather than dictionary overwrite.
- [x] Implement normalized `BoutAuthorityIdentity` comparison.
- [x] Publish only the roster-selected core bout table; require the maintained
      chaser descendants to bind that exact bout dependency, so no second bout
      authority reaches composition. Retain the normalized comparator for
      explicit legacy migration/comparison rather than invoking it on an
      unselected source.
- [x] Reject duplicate table names and competing authority bindings at plan
      time.
- [x] Add exact foreign-key/cardinality contracts from paradigm rows to core
      rows.
- [x] Reproject chaser motion/body relationships against selected core rows;
      do not copy `provider_motion_samples` or `body_frame_samples` into the
      composite as competing core facts.
- [x] Preserve the existing `validated_behavior/v1` manifest family and generic
      publisher/reader; use a new installed profile ID rather than changing
      Phase C in place.
- [x] Add real writer-to-unpatched-reader boundary coverage.
- [ ] Prove one representative legacy/core bout pair equal or conflict through
      the normalized comparator. This is migration evidence, not an admission
      prerequisite for the maintained direct-core composite.
- [ ] Validate one selector-ineligible canary before any maintained use.

### Track D — migrate maintained paradigm consumers

- [x] Replace composable chaser `MOTION_BOUT_PAIRS`/`matches[0]` authority
      selection with the persisted core roster.
- [x] Make chaser body-relative, bout-response, epoch, escape/freeze,
      occupancy, and response products consume selected core identities.
      Direct and transitive publications now preserve the exact sealed roster
      dependency; paired-provider products require one common identity.
- [ ] Add the core-roster dependency to shared stimulus response so moving
      grating, concentric/radial OMR, and looming cannot resolve independent
      latest motion or bout runs.
- [x] Remove implicit `track_id=0` from the maintained chaser composite
      dependency path;
      require the selected track identity and cardinality.
- [ ] Point activity/spatial bins and classifier-specific extensions at the
      shared roster while retaining their genuinely distinct output grains.
- [ ] Add static CI ratchets against `latest`, ordered-first, implicit-track,
      and fallback source selection in maintained composite planners.
- [ ] Keep archived Phase-C and explicitly legacy direct-export paths readable;
      do not silently reinterpret them as roster-bound outputs.

## Acceptance criteria

A future composite implementation is acceptable only when:

- its manifest contains one exact core authority roster per recording;
- every paradigm table binds that roster and its own paradigm authority;
- no duplicate `canonical_swim_bouts` table or unresolved bout source exists;
- bout equality is established through normalized scientific identity and
  selected-event content, not shared table schema, field spelling, or run name;
- overlapping motion/body fields are either published once or explicitly
  documented and equality-validated projections;
- profile composition detects collisions before constructing the table map;
- core-only, chaser-extended, and other paradigm profiles all use the same
  planner, atomic publisher, manifest, validator, and reader;
- no maintained composite dependency is chosen by `latest`, ordered-first
  match, implicit track zero, or fallback after a selected-profile failure;
- unsupported or conflicting profiles return typed blocked plans without
  scratch writes;
- a real sealed Phase-C publication remains readable by the new reader, an old
  reader fails closed on the new profile ID, and a real composite publication
  is readable through the unpatched generic reader;
- receipt-backed tail and subject-shape loading performs no decoded hash reads
  merely to re-prove payload identity, and receipt-free maintained loading
  fails before scientific array traversal; and
- required CI and a real selector-ineligible publication canary pass.

## Review record

Six independent Luna xhigh read-only investigations reviewed checkpoint
`afbc1d0d6af822ca7cc4e3b051cdd9bc981df80c` on 2026-09-03:

1. Phase-C/core table overlap and subtraction;
2. shared resolver, bundle, profile, publisher, and reader architecture;
3. canonical swim-bout identity and conflict behavior;
4. chaser, moving-grating, OMR, looming, and generic core-consumer inventory;
5. immutable Phase-C compatibility and profile/manifest versioning; and
6. the tail receipt-validation hot path observed in the real-source canary.

All six agreed on the central decision: use one core authority roster, one
composite bundle set, one installed profile, and the existing
`validated_behavior/v1` engine; preserve Phase C unchanged; treat paradigm
outputs as extensions; and fail closed on competing core identity. The reviews
also independently identified the maintained-path selection problems and the
absence of a supported tail payload receipt in the inspected canary.

The reviews did not edit files, run tests, mutate Git state, or touch scientific
publications. Their static findings are incorporated above. They are design and
implementation evidence, not substitutes for the pending real-source canaries,
boundary tests, and required CI.

## Non-goals

This decision does not:

- require every recording to have a paradigm extension;
- prohibit multiple candidate analysis runs from coexisting in a Zarr;
- declare one projection schema universally correct for every query grain;
- treat Parquet as the scientific authority;
- mutate or republish the completed GoodBatBadBat Phase-C generation;
- authorize heuristic source selection or fallback; or
- claim that the completed static overlap audit is runtime authority proof for
  any particular legacy/core source pair.
