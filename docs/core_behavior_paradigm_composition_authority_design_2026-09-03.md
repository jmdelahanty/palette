# Core behavior and paradigm export authority design — 2026-09-03

<!-- decision-meta
status: accepted-design-current-implementation-partial
created: 2026-09-03
last_updated: 2026-09-03
baseline_commit: 07db267c
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
also identifies a concrete performance question for Track B: determine whether
the immutable tail publication's sealed array/measurement receipts can satisfy
this read boundary without replaying the full decoded hash. Any optimization
must preserve exact authority validation and fail closed for absent, stale, or
unsupported receipts.

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

## Implementation plan

### Track A — finish the complete core base

- [ ] Finish the five-grain source resolver and recording-bundle schema.
- [ ] Require one exact authority for every required core capability.
- [ ] Seal the cross-grain join record and bind every capability to it.
- [ ] Reuse one strict source binding within a recording to avoid repeated
      whole-source validation.
- [ ] Keep the five scientific grains in separate normalized tables.
- [ ] Route planning, sharding, publication, validation, and reading through
      the existing generic cohort engine.
- [ ] Prove that no new publisher, selector, manifest family, or CLI path was
      introduced.
- [ ] Add real execution-report-to-resolver and generic-publisher boundary
      tests.
- [ ] Run a read-only admission canary for all four Sleepyfish cameras.
- [ ] Run required CI before merge, deployment, or production publication.

### Track B — audit paradigm composition

- [ ] Inventory the exact authority bindings and table grains in Phase C.
- [ ] Classify every Phase-C table as core fact, paradigm extension,
      denormalized convenience projection, metadata, or summary.
- [ ] Compare `provider_motion_samples` with `kinematics_samples` field by
      field and identify the reusable base versus chaser-only additions.
- [ ] Compare `body_frame_samples` with `subject_body_frame_samples` including
      row, timing, and validity semantics.
- [ ] Prove whether Phase-C and core-workflow bouts can bind the same source
      identity on representative recordings.
- [ ] Specify the minimal chaser extension table roster after core-table
      subtraction.
- [ ] Inspect moving-grating and other maintained workflows for independent
      motion/body/bout selection that should instead consume the core roster.
- [ ] Propose compatibility and versioning rules without rewriting Phase C.

### Track C — implement a composite only after Track B

- [ ] Register one explicit composite profile on `validated_behavior/v1`.
- [ ] Compose the complete core specs with paradigm-only extension specs.
- [ ] Reject duplicate table names and competing authority bindings at plan
      time.
- [ ] Add exact foreign-key/cardinality contracts from paradigm rows to core
      rows.
- [ ] Add real writer-to-unpatched-reader boundary coverage.
- [ ] Validate one selector-ineligible canary before any maintained use.

## Acceptance criteria

A future composite implementation is acceptable only when:

- its manifest contains one exact core authority roster per recording;
- every paradigm table binds that roster and its own paradigm authority;
- no duplicate `canonical_swim_bouts` table or unresolved bout source exists;
- overlapping motion/body fields are either published once or explicitly
  documented and equality-validated projections;
- core-only, chaser-extended, and other paradigm profiles all use the same
  planner, atomic publisher, manifest, validator, and reader;
- unsupported or conflicting profiles return typed blocked plans without
  scratch writes; and
- required CI and a real selector-ineligible publication canary pass.

## Non-goals

This decision does not:

- require every recording to have a paradigm extension;
- prohibit multiple candidate analysis runs from coexisting in a Zarr;
- declare one projection schema universally correct for every query grain;
- treat Parquet as the scientific authority;
- mutate or republish the completed GoodBatBadBat Phase-C generation;
- authorize heuristic source selection or fallback; or
- implement the full five-grain-plus-chaser composite before the overlap audit
  is complete.
