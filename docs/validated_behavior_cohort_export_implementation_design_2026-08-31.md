# Validated behavior cohort export design and implementation checklist

<!-- decision-meta
status: implementation-ready-design
created: 2026-08-31
last_updated: 2026-08-31
baseline_commit: 32150c90c376c0bcc398e0c7d141ace33515a4f9
scope: immutable cohort membership, recording-bundle fanout, normalized
  cross-recording Parquet export, lazy consumers, validation, and staged
  GoodBatBadBat adoption
related: docs/validated_recording_behavior_composition_design_2026-08-31.md,
  docs/cohort_release_workflow.md,
  docs/dataset_reporting_contract.md,
  docs/diagnostics/goodbatbadbat_subject_identity_reuse_2026-08-18.md,
  docs/diagnostics/chaser_exact_full_gap_closure_implementation_checklist_2026-08-30.md
-->

## Question this design answers

Palette now has a recording-local validated behavior bundle that closes one
exact set of Core Behavior and chaser sources without copying their numerical
payloads. The next question is:

> How should Palette export a cohort of those bundles as one portable dataset
> without erasing row grains, capability failures, provider identity, source
> lineage, or the distinction between descriptive samples and independent
> experimental units?

The answer is not one wide table and not a copied cohort Zarr. It is one
manifest-selected immutable Parquet dataset made of normalized tables, one
validated recording shard at a time. Recording Zarr arrays remain the
scientific authorities. The export is a query and interchange surface whose
manifest closes exact cohort membership, bundle identity, table contracts,
part inventory, and capability coverage.

## Decision

1. Publish a **validated behavior cohort export** as an immutable collection of
   normalized Parquet tables plus one manifest and validation receipt.
2. Keep each scientific row grain in its own table. Never form a wide join of
   frames, chasers, trials, bouts, events, epochs, and bins.
3. Freeze the parent cohort before export. Preserve all parent members and give
   every member an explicit export disposition, including invalid or
   unavailable members.
4. Require exactly one validated recording-behavior bundle for every admitted
   recording. A bundle path, file digest, record digest, and its exact child
   bindings are part of the exported source identity.
5. Represent capability state per recording. A capability present in one
   recording cannot make that capability appear cohort-wide.
6. Export only persisted scientific facts or exact bounded projections allowed
   by their source contract. Do not reconstruct unpersisted visits, aligned
   trajectories, response regimes, or legacy distance summaries during export.
7. Preserve position-provider identity as a first-class dimension. Keypoint
   and detection position rows are never silently averaged, substituted, or
   interpreted as anatomical equivalents.
8. Use `recording_id` as the temporary recording-by-animal analysis unit for
   this historical cohort, exactly as authorized by the subject-identity
   incident decision. Retain the reused acquisition UUID only as provenance;
   it is not a grouping key or corrected subject authority.
9. Treat missing historical acquisition-batch identity as missing. Never infer
   a batch from timestamps, paths, recording names, arenas, or adjacency.
10. Build source shards independently, validate them, fan in only successful
    receipt-backed shards, validate the complete dataset, and commit the
    immutable publication manifest last.
11. Keep the first publication selector-ineligible, non-production, and out of
    active registry selection. Promotion is a separate governed decision after
    required CI and cohort validation.

## Generic composition boundary

The cohort/export format is not a GoodBatBadBat or chaser-specific formula.
It has three intentionally separate layers:

1. The generic core owns immutable membership, analysis-unit identity, bundle
   state, a profile-declared capability matrix, table-contract registration,
   shard inventory, publication, validation, and lazy reading. It contains no
   GoodBatBadBat recording IDs, cohort counts, chaser capability names, or
   protocol-specific numerical formulas.
2. Source adapters normalize a particular frozen roster and recording-bundle
   profile into that core. The first adapters understand the historical
   composable-chaser task v5, future frozen-cohort v2, exact-chaser projection
   receipts, and validated-recording-behavior bundle v1. These are boundary
   translators, not alternate exporters.
3. Scientific table adapters are registered by explicit table contracts and
   required capabilities. A new protocol or analysis family adds a small
   source/table adapter; it reuses the same shard, validation, publication, and
   consumer machinery.

The 84/80/4 GoodBatBadBat counts, four invalid recording IDs, protocol hashes,
temporal-proxy policy, and subject-identity incident are therefore manifest
data. They must never appear as branching constants in the generic exporter.
Likewise, a capability contract declares its exact sorted keys and typed state
and reason vocabulary. The bundle-set envelope does not assume that every
validated behavior profile has gaze, escape, radial, feeding, or any other
specific capability.

## Non-goals

This work does not:

- recompute or rewrite any recording Zarr scientific payload;
- repair subject UUIDs or mint biological identities in Palette;
- infer missing acquisition batches;
- promote shape, eye, gaze, or tail candidates merely because their groups
  exist;
- convert trial-envelope gaps into trial members;
- introduce interpolation, nearest-timestamp attachment, forward filling, or
  row-order joins;
- describe the historical controller-input provenance proxy as verified
  physical stimulus presentation or camera exposure;
- revive the old unsealed chaser distance-summary or distance-histogram path;
- create individual near-field visit rows from aggregate visit statistics;
- create event-aligned escape trajectories from event summaries;
- pool frame, chaser, bout, event, or bin rows as independent analysis units;
  or
- update selectors, the production registry, or the mutable shared `/groups`
  checkout.

## Audited GoodBatBadBat starting point

The implementation baseline is merge commit
`32150c90c376c0bcc398e0c7d141ace33515a4f9`, which contains the recording-local
validated behavior bundle and exact bundle-backed provider-motion adapter.

The closest exact 84-recording source task is:

```text
/groups/johnson/johnsonlab/jeremy/operations/
  goodbatbadbat_chaser_body_frame_20260827/full_cohort_runs/
  composable_chaser_successors_epoch_alignment_recipe_v2_full84_
  a03076b5_c0ec5dc0_20260830/cohort_task.json
```

Its schema is `palette.composable_chaser_successor_cohort_task` version 5, its
file SHA-256 is
`94d01be7c1fd1941ed0f868b51c3dbf037a03d8fc8935c5c66a3c01f56dee1c2`, its
declared `task_sha256` is
`a03076b5c18aa10b32de60d8b6c50c9edd09584eb131b6e1412525be6fb4a84b`,
and it contains 84 deterministically ordered recording entries. The task binds
an historical registry snapshot hash, but the snapshot path is under `/tmp`.
The new durable membership manifest must therefore carry the exact normalized
roster and its own digest instead of depending on that ephemeral path at read
time.

Current exact chaser receipt coverage is:

| Parent cohort state | Recording count | Export disposition |
|---|---:|---|
| Exact successor and projection-receipt closure | 80 | Eligible for bundle planning and compact scientific tables |
| Invalid overlapping or non-strictly ordered raw semantic step bounds | 4 | Retained in membership and capability tables; excluded from bundle-backed scientific tables |
| Total parent cohort | 84 | Denominator for membership and coverage reporting |

The four invalid recordings are:

- `2026-08-12T21-14-36Z_arena_1_goodbatbadbat`
- `2026-08-12T21-14-36Z_arena_2_goodbatbadbat`
- `2026-08-12T21-14-36Z_arena_3_goodbatbadbat`
- `2026-08-12T21-14-37Z_arena_4_goodbatbadbat`

Their observed failure is
`ProtocolSemanticChaserSelectionError: Raw semantic step bounds overlap or are
not strictly ordered.` This is invalid evidence, not an empty recording and not
a reason to remove the recording silently.

The full-84 task currently has 80 complete receipts, each marked
`complete_selector_ineligible` and produced from the `c0ec5dc0...` commit.
Production authority, selector eligibility, selector activation, and registry
update are false. The receipt tree contains 81 projection-receipt files but only
80 unique recordings because the Arena-1 canary has both an older and a current
receipt generation. File count must therefore never stand in for recording
count.

One durable recording bundle exists for
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`:

```text
/groups/johnson/johnsonlab/jeremy/operations/
  goodbatbadbat_validated_recording_behavior_bundle_20260901_v001/
  2026-08-10T17-20-55Z_arena_1_goodbatbadbat_
  validated_recording_behavior_bundle_v001.json
```

Its file SHA-256 is
`eac38edf794941aa9b23cf463b9ee2074df58c766b792e2d9826ca18222855cb`
and its internal record digest is
`d798d7f86a35ce054dc61294772b1938a39538adf287a6d22026eeb99d39e52f`.
It binds the older `c5eebd839a2fd277322889f388a950d029e47db7` projection
receipt generation. The coherent 80-recording receipt set is under
`c0ec5dc0310db4799b09a6fd96c041f32af99a68`. The cohort run must regenerate
the canary bundle against the deliberate `c0ec...` receipt rather than mix
receipt generations merely because their child payload digests happen to
match.

The recording-local composition design predates this post-merge materialized
file and still describes its own earlier `/tmp` canaries as non-durable. That
historical statement remains true for those temporary files; it does not
invalidate the durable artifact above, whose envelope records commit
`32150c90c376c0bcc398e0c7d141ace33515a4f9`, 17 complete capabilities, four
typed unavailable capabilities, and the full selector-ineligible safety
envelope.

Run-name heterogeneity is not itself a scientific mismatch. The current source
set contains canary and ordinary run names. Admission depends on exact schema,
method, provider, coordinate, row-axis, track, and child-lineage compatibility;
it must not require string-equal run names when their contracted methods are
compatible. Conversely, equal names or shapes never prove compatibility.

The existing full-84 provider-epoch Parquet publication is useful diagnostic
evidence but is not this export's authority. It is selector-ineligible and
explicitly `linear_only`; angular/heading metrics were excluded because most
recordings used a superseded heading cache. The new export reads the strict
bundle-bound semantic-v2 child and preserves that child's exact metric
disposition. Motion heading and anatomical body heading remain separately
identified quantities.

## Identity and denominator contract

The operator-confirmed incident record states that all 84 recordings represent
distinct animals, while eight capture-time subject UUIDs were reused across 16
recordings. The bound incident file SHA-256 is
`94c5e2e3daf74a90cad3f1c7cac66cac43094b15a007a345474da443ddb97d72`.
For this urgent historical analysis:

- the parent cohort denominator is 84 recording-scoped analysis units under
  the operator decision;
- the currently chaser-admissible subset is 80 recording-scoped analysis
  units;
- `recording_id` is the authorized analysis-unit key;
- `source_subject_id` may be retained for audit only;
- `source_subject_id` must never be used for grouping, balancing, joining, or
  sample-size calculation;
- `subject_identity_status` must state
  `capture_uuid_reuse_incident_recording_scoped_workaround`;
- `subject_identity_decision_path`, its file SHA-256, and the operator decision
  timestamp `2026-08-18T12:31:57Z` must be bound by the membership and export
  manifests; and
- `acquisition_batch_id` remains null with status
  `missing_historical_not_inferred` unless an authoritative successor later
  supplies it.

The workaround is not a UUID correction. Any table, figure, or statistical
result using it must say that its independent unit is recording-scoped. A
future MetaZebrobot-backed identity successor may supersede the incident-bound
identity surface through a new export generation; it must not mutate this one.

Schema v1 additionally admits exactly one bundle-selected track for each
recording-scoped unit. `track_id` remains a source/entity key and never
replaces `recording_id` as the independent unit. A recording with multiple
admitted tracks requires an exact track-to-subject/experimental-unit authority
and a later contract version; it cannot enter this v1 export by creating
multiple apparent units.

The current primary registry has no authoritative GoodBatBadBat identity rows,
and its generic analytics identity receipt would treat the reused subject UUID
as the experimental unit. This export must not fabricate fields to satisfy that
contract. Its incident-bound recording-unit identity is a separate, explicit
historical profile. Future corrected subject/setup successors can create a new
canonical identity-bound export without rewriting this generation.

## Architectural boundary

```text
exact parent cohort task / future frozen cohort
                         |
            durable membership manifest (84)
                         |
       +-----------------+------------------+
       |                                    |
  80 exact projection receipts         4 invalid members
       |                                    |
  recording-bundle fanout           explicit no-bundle states
       |                                    |
  80 validated bundles (target)              |
       +-----------------+------------------+
                         |
             bundle-set manifest (84 rows)
                         |
       per-recording, per-table shard fanout
                         |
          all-required-success receipt barrier
                         |
       deterministic table inventory and fan-in
                         |
             cohort-specific deep validation
                         |
      atomic immutable Parquet publication
                         |
       manifest-selected lazy reader / adapters
                         |
        Marimo, static figures, and statistics
```

The bundle-set and export manifests are composition authorities. They do not
replace any child scientific authority and do not make the Parquet copy more
authoritative than its source Zarr.

## Required contracts

### Durable cohort membership manifest

Introduce schema
`palette.analysis.validated_behavior_cohort_membership`, version 1. It is the
only membership input accepted by the bundle-set planner and export planner.

For this historical cohort, one explicit migration command may build it from
the exact schema-v5 task. Future cohort creation should build the same interface
from `palette.frozen_cohort_manifest` version 2. Export consumers do not learn
multiple historical membership formats.

Required top-level fields:

- schema ID/version, membership ID, creation time, and self digest;
- exact builder software commit;
- source membership schema/profile, path, file digest, and declared internal
  digest;
- normalized parent-member count and deterministic ordering policy;
- canonical digest of the sorted member records;
- subject-identity incident path, file digest, decision timestamp, and approved
  analysis-unit policy;
- missing acquisition-batch policy;
- allowed disposition/state vocabulary; and
- safety flags: selector ineligible, non-production, no registry update, and no
  source mutation.

Each of the 84 member records must contain:

- stable ordinal;
- `dataset_id`, `recording_id`, and canonical absolute analysis-Zarr path;
- protocol name/hash and the exact source-task entry digest;
- raw source subject UUID when available, plus its non-authoritative status;
- null or authoritative acquisition-batch identity plus identity status;
- `analysis_unit_kind` and `analysis_unit_id`;
- membership state: `admitted`, `invalid`, `unavailable`, or `excluded`;
- typed reason code and evidence for every non-admitted state; and
- the exact intended projection-receipt path/file digest when admitted.

`dataset_id` is the stable dataset identity. The absolute Zarr path is the
locator captured at selection and must be canonical and root-confined, but path
text alone is not scientific identity. A relocation requires an audited locator
successor or a newly frozen membership generation; path normalization cannot
silently change the bound source.

The validator must prove 84 unique recording IDs, 84 unique analysis-unit IDs,
one canonical Zarr per recording, deterministic ordering, exact equality to the
bound parent roster, and an explicit disposition for every parent member.

### Recording bundle-set manifest

Introduce schema
`palette.analysis.validated_recording_behavior_bundle_set`, version 1. It binds
the membership manifest to exactly one bundle state per member.

Each member record must include:

- membership ordinal and member-record digest;
- recording and analysis-Zarr identity;
- bundle state and typed reason;
- for a complete bundle: absolute bundle path, file SHA-256, internal record
  SHA-256, schema/method/status, exact projection-receipt path and digest, and
  canonical digest of all source and child bindings;
- for a missing/invalid bundle: no fabricated path or digest and a state that
  agrees with the membership disposition;
- one typed state and reason for every closed capability key; and
- a member capability-record digest.

The top-level manifest must contain canonical aggregate digests for member
bindings and the complete member-by-capability matrix. It must reject duplicate
recordings, extra bundles, missing member records, selector-named paths, source
substitution, and a bundle whose recording/Zarr/receipt identity disagrees with
membership.

### Per-recording export shard receipt

Introduce schema
`palette.analytics.validated_behavior_export_shard`, version 1. One shard owns
one recording and may contain zero or more table parts. It must bind:

- export-plan ID/digest and member ordinal;
- membership and bundle-set manifests/digests;
- recording, Zarr, bundle, projection-receipt, and source-binding digests;
- exact requested table contract IDs/versions and capability policy;
- software commit and commit-pinned deployment path;
- deterministic input and output parameters;
- one inventory record per part: relative path, byte size, row count, file
  SHA-256, Arrow schema digest, table contract, and primary-key bounds;
- explicit zero-row justification where the table contract permits zero rows;
- validation status and validation-policy version; and
- receipt self digest.

Workers write only to unique staging or scratch paths. A shard is reusable only
when every bound input, parameter, contract, software commit, and expected part
digest is identical. A same-name mismatch fails; it never overwrites or
silently resumes.

### Cohort export manifest and validation receipt

Introduce schema `palette.analytics.validated_behavior_cohort_export`, version
1. Reuse the existing immutable publication inventory and exact Arrow contract
mechanisms, while extending them with:

- membership and bundle-set paths/digests;
- parent, admitted, invalid, unavailable, and excluded recording counts;
- the full per-recording capability matrix or its exact digest plus a selected
  manifest table containing the matrix;
- table-specific capability policies and exact contributing member sets;
- deterministic ordered shard receipt roster and aggregate digest;
- table contracts, Arrow contracts, grains, primary keys, foreign keys, units,
  providers, coordinate systems, methods, parameters, and source lineage;
- exact part inventory, part/file hashes, row counts, and global key ranges;
- requested and effective Parquet partition/row-group policy;
- identity-incident and missing-batch policies;
- temporal-alignment requirement/class, physical-presentation verification,
  presentation timestamp/clock/exposure availability, proxy selection policy,
  native-state multiplicity policy, and scientific-use class;
- publication generation and manifest-last atomic commit evidence;
- validation receipt path/digest; and
- safety flags that remain false for selector eligibility, production
  authority, automatic registry indexing, and selector activation.

The cohort validation receipt must re-open only manifest-selected parts. It
must never glob a table directory or resolve a `latest` export.

## Capability policy

Retain the recording-bundle states exactly:

- `complete`
- `unavailable`
- `inapplicable`
- `invalid`
- `stale`
- `review_required`

Every requested table declares one of these policies:

| Policy | Meaning |
|---|---|
| `required_all_admitted` | Every admitted recording must have the capability complete; otherwise the export fails |
| `optional_explicit_coverage` | All parent members remain in capability coverage; scientific rows come only from complete members |
| `capability_stratified_subset` | Publish a named immutable child subset with its exact complete-member roster and denominator |

No table may infer availability from the union of columns or parts. Zero rows
are not a substitute for unavailable, invalid, stale, review-required, or
inapplicable state.

A complete capability may legitimately yield zero events. In that case the
table is declared with its exact empty Arrow schema, zero row count,
contributing recording, and typed zero-row reason; the reader returns an empty
lazy frame. An unavailable capability omits scientific rows and is represented
in `recording_capabilities`. These cases are never conflated.

For the first GoodBatBadBat export, recording membership and capability
coverage are required for all 84 parent members. Bundle-backed scientific
tables use the exact 80-member complete set unless a stricter table-specific
capability policy produces a smaller explicit subset. Gaze, eye-angle,
subject-shape, and tail data remain absent when the bundle records them as
unavailable, regardless of similarly named candidate groups in the Zarr.

Candidate subject-shape, eye-angle, and gaze prerequisite payloads do exist for
the cohort. Their current receipts are selector-ineligible; human biological
gaze-direction acceptance is false or pending, body-component review/QC is not
accepted as an authority, and the gaze-convention review remains pending. This
is candidate evidence, not an admissible gaze capability.

For each of the four non-admitted recordings, semantic selection is `invalid`
with the exact overlapping-bounds reason. Bundle-dependent capabilities are
`unavailable` with `blocked_by_invalid_semantic_selection` or another exact
upstream reason. Otherwise present candidate groups are not upgraded to
complete because no validated recording bundle closes them.

This does not assert that all Core Behavior evidence in those four recordings
is scientifically invalid. It says the current chaser-backed bundle schema
cannot close it independently of the invalid semantic source. If those four
recordings must later contribute base-only rows, define a separate exact
core-only bundle profile and capability-stratified export; do not weaken this
bundle or bypass it from the exporter.

## Normalized table plan

All table primary keys begin with `recording_id` unless the row is explicitly a
cohort-level manifest fact. Every scientific table also carries the export run
ID, membership member digest, bundle record digest, exact source-child digest,
and provider identity needed for its rows.

Each table has the exact contract ID
`palette.analytics.table.<table_name>` and an explicit contract version. The
validated-behavior export envelope has its own schema ID and does not masquerade
as the existing generic `palette.analytics_export` version 3 family.

### Phase A: compact source and scientific tables

| Table | Grain and key suffix | Source | Initial status |
|---|---|---|---|
| `cohort_recordings` | one row per parent recording | membership manifest | Required; 84 rows |
| `recording_bundles` | one row per parent recording and bundle state | bundle set | Required; complete bundle identity for 80 and explicit non-complete state for four |
| `recording_capabilities` | recording x capability | bundle set | Required; explicit state/reason |
| `recording_source_bindings` | recording x source binding | complete bundle | Required for the 80 complete bundles |
| `position_providers` | recording x exact position provider | complete bundle and child manifests | Implement keypoint and detection as distinct providers |
| `chaser_occurrences` | recording x stimulus-scoped chaser occurrence | relative/semantic children | Implement identity and behavior role independently of display style |
| `semantic_epochs` | recording x semantic window | semantic-selection child | Implement |
| `controller_trials` | recording x chaser x logged trial | controller child | Implement |
| `canonical_swim_bouts` | recording x track x bout | exact same-track swim-bout source | Implement independently of stimulus |
| `bout_chaser_associations` | recording x bout x chaser | generalized bout-response child | Implement; do not duplicate canonical bout facts |
| `bout_response_distance_bins` | recording x epoch/role x chaser x persisted distance bin | generalized bout-response child | Implement persisted summaries only |
| `trial_escape_freeze_events` | recording x source bout-chaser event | trial-locked escape/freeze child | Implement event facts, chaser identity, and censor/trace-valid fields |
| `trial_escape_freeze_summaries` | recording x chaser x controller trial | trial-locked escape/freeze child | Implement persisted summaries |
| `trial_escape_freeze_threshold_sweeps` | recording x persisted threshold-sweep row | trial-locked escape/freeze child | Implement as its own grain; never join back as independent events |
| `epoch_behavior_summary` | recording x semantic epoch | semantic-v2 epoch child | Implement |
| `spatial_occupancy_support` | recording x provider x epoch | spatial-occupancy child | Implement candidate, valid, in-arena, invalid, and out-of-arena denominators |
| `spatial_occupancy_bins` | recording x provider x epoch x x-bin x y-bin | spatial-occupancy child | Implement persisted bin counts/fractions plus grid-recipe identity |
| `radial_near_field_summary` | recording x provider x epoch x chaser | radial child | Implement aggregate entry/dwell/coverage facts |
| `same_quadrant_occupancy` | recording x provider x epoch x chaser | radial child | Implement persisted valid- and candidate-denominator scalar facts |
| `radial_near_field_density_bins` | recording x provider x epoch x chaser x radial bin | radial child | Implement persisted bins |
| `radial_near_field_distance_cdf` | recording x provider x epoch x chaser x threshold | radial child | Implement persisted thresholds |
| `body_alignment_distance_bins` | recording x epoch x chaser x distance bin | body-alignment child | Implement; provider and anatomical-body authorities explicit |

`same_quadrant_occupancy` is a separate normalized table even though its scalar
facts are sealed by the same exact radial child. `radial_near_field_summary`
retains behavior role, geometry authority, wall-excluded support, geometric-
null expectations, expected-count suppression, visit-policy identity, and
every numerator/denominator field. Geometric enrichment is a position-relative
control, not by itself behavioral evidence. Near-field visit hysteresis is not
interchangeable with the wider bout-response traversal and shell thresholds.

The bundle-backed `chaser_escape_freeze` v2 child is a trial-locked diagnostic
and uses table IDs prefixed `trial_escape_freeze_*`. It is distinct from the
older `palette.chaser_escape_events.v3` component under the former chaser-
distance family. That older component is not admitted by this bundle. Any
future comparative escape-event successor uses its own event x reference grain
and cannot be substituted into the trial diagnostic tables.

The implementation should reuse existing semantic table contracts only where
their grain, source authority, identity requirements, and units are exactly
compatible. An old table name does not authorize reading an old source. In
particular, existing generic chaser exports that depend on the earlier
`analysis/chaser_distance_runs` authority cannot be redirected to a successor
with different semantics without a new contract version.

Provider selection is table-specific and manifest-bound:

| Surface | Current GoodBatBadBat provider policy |
|---|---|
| Provider motion and canonical bouts | Exact bundle-selected keypoint-triad motion/track |
| Relative position and simple distance | Separate keypoint and detection providers |
| Radial/near-field summaries | Separate keypoint and detection children |
| Spatial occupancy | Paired provider output with provider retained in every key |
| Body alignment and bearing | Bundle-bound keypoint position plus the exact anatomical body-frame authority |
| Bout response and trial escape/freeze | Exact provider named by the child lineage; never substituted at export time |

This policy is a frozen choice for this cohort, not universal provider
precedence. Detection/bounding-box centroids remain a first-class position
provider; they are not an anatomical body-frame supplier.

### Phase B: optional high-volume exact projections

| Table | Grain | Admission rule |
|---|---|---|
| `provider_motion_samples` | recording x provider x track sample | Exact bounded projection from the bundle-bound provider-motion run |
| `stimulus_native_state_support` | recording x temporal-proxy binding x acquisition frame x chaser x native stimulus state | Every contributing state key/frame/source row/timestamp and exact multiplicity; no display-exposure claim |
| `chaser_relative_samples` | recording x provider x acquisition frame x chaser | Exact relative-frame child; replaces any temptation to revive unsealed legacy distance tables |
| `body_frame_samples` | recording x acquisition frame | Exact bundle-bound anatomical body frame; kept separate from chaser repetition |
| `body_relative_samples` | recording x acquisition frame x chaser | Exact persisted body-alignment frame rows without treating repeated body values as independent samples |
| `controller_trial_membership` | recording x acquisition frame x chaser/trial | Exact logged active membership only; gaps stay separate |
| `controller_trial_gap_evidence` | recording x acquisition frame x chaser/trial envelope | Exact persisted gap state/reason; never trial membership |

These tables require streaming or per-recording Parquet writing. They must not
return all rows to one parent process. Their row-group policy is a measured and
recorded storage decision, not a hidden library default.

### Explicitly unavailable until new successors exist

| Requested surface | Why it is unavailable |
|---|---|
| Individual near-field visits | Current radial successor persists aggregate entry counts, dwell summaries, radial bins, and CDF thresholds, not one durable row per visit |
| Ring-entry static panels and source-frame media | Depend on individual visit identity; media additionally requires exact video/frame mapping |
| Event-aligned escape-distance trajectories | Escape/freeze v2 persists event facts, recapture status/latency, and trace validity, but not aligned sample rows |
| Response regimes v2 | No exact current successor is bound by the bundle; the older compatibility implementation is not promoted into this export |
| Full quadrant-joint occupancy matrix | Provider computation exists, but the current exact radial child does not seal the complete matrix as an export authority |
| Epoch wall/center behavior from the epoch child | The semantic-v2 summary intentionally omits spatial facts; a distinct exact position-and-geometry-bound epoch-spatial successor is still required |
| Chaser-tail response | Tail is not yet bound to an exact chaser event/alignment successor |
| Gaze cohort tables for this batch | Bundle capability is unavailable because accepted segmentation/review evidence is absent for the intended batch |
| Legacy chaser epoch distance summary/histogram | The old semantics were not independently sealed; export the exact relative samples or successor-persisted bins instead |
| Full distance x bearing histograms and unrestricted epoch bearing summaries | The exact body-alignment child persists frame rows and distance-conditioned bearing summaries, but not the old generic egocentric histogram contract |
| Full-profile/module readiness | Recording capability state is not a substitute for the separately governed full-profile applicability envelope, which is not yet bundle-integrated |
| Bundle-backed baseline summary | The exact bundle route does not yet bind a baseline-summary capability; canonical bouts and semantic-v2 epoch behavior remain available |

Whole-session speed, heading, and position are available only through the
optional exact `provider_motion_samples` projection in Phase B. Phase A does
not invent a compact full-session aggregation. The exact
`body_alignment_distance_bins` table supplies persisted bearing/alignment-by-
distance summaries; a full bearing polar distribution requires the Phase-B
frame rows or a future independently contracted summary.

Adding one of these surfaces requires a new scientific successor and bundle
capability, then a new export contract version or additive table contract. It
is not an exporter convenience transformation.

## Exact join rules

1. Recording joins require exact recording ID and canonical analysis-Zarr
   identity.
2. Frame joins require exact acquisition-frame and track-row identity proved
   compatible by the bundle. Same length is insufficient.
3. Timestamp joins are exact only when both sources bind the same session-time
   authority. There is no nearest-neighbor matching.
4. Chaser joins require exact chaser identity codes and their digest-bound
   registries. Array order and display color are not identity.
5. Provider-specific rows retain `position_provider_id`, provider digest, and
   provider role. Detection centroids remain a first-class position provider
   but cannot supply anatomical body orientation.
6. Epoch membership uses exact semantic window identity and membership arrays.
   Rows outside semantic epochs remain source evidence but are not assigned to
   an epoch.
7. Trial events attach only to exact logged active-member rows. Rows in the
   first-to-last trial envelope that lack active membership are retained as gap
   evidence and are not trial members.
8. Bouts remain canonical stimulus-independent facts. Chaser, trial, epoch,
   and response data live in association tables keyed back to the canonical
   bout.
9. Escape events retain invalid/censored trace states; event counts are not
   dropped merely because recapture evidence is unusable.
10. Scientific bins use the exact persisted edges, bin policy, denominator,
    and source recipe. Export or viewers may not silently re-bin contracted
    summaries.
11. Every semantic interval records its exact boundary convention and source
    interval digest. The current GoodBatBadBat selection is half-open
    `[start_frame, end_frame)`. Do not substitute textual legacy
    `black_before`/`black_after` labels for exact pre/training/post roles.
12. Relative rows retain fish and chaser source-row IDs plus the bound temporal
    projection policy. The manifest records native-to-acquisition multiplicity
    diagnostics so a relation-row count cannot be misreported as camera-frame
    count or biological sample size.
13. Canonical bouts use exactly the bundle-selected default signal and signal
    level. Multiple measurement levels cannot be exported as multiple physical
    bouts; optional level-specific metrics require a measurement-level key.
14. Every spatial table retains transform direction, coordinate authority and
    digest, reviewed-arena identity, grid/zone policy, normalization ID, and
    exact support denominator. Misleading historical helper names cannot
    override the contracted camera-to-analysis transform direction.

For these historical recordings, every affected table and the export manifest
must literally retain:

```text
temporal_alignment_requirement = input_provenance_proxy_allowed
temporal_alignment_class = controller_input_provenance_proxy
physical_presentation_verified = false
presentation_timestamp_available = false
camera_presentation_clock_transform_available = false
camera_exposure_reference = unknown
scientific_use_class = exploratory_proxy
```

There is no fallback from `physical_presentation_required`. The Phase-B native
support relation retains `stimulus_state_key`, `stimulus_frame_num`, native
timestamp/source-row identity, acquisition mapping, all contributing native
IDs, and multiplicity. The acquisition-frame proxy row retains its selected
source row and policy. Neither row is called the stimulus displayed during the
camera exposure.

For the current cohort, the semantic-v2 epoch source records
`physical_speed_level = filtered`. Every speed field must retain that exact
level and source identity. An unlabeled or silently substituted raw speed is
inadmissible. Time fields also remain distinct: acquisition-frame identity,
session timestamp nanoseconds, and derived seconds are never interchangeable.
Physical distances are calibrated millimetres; pixel positions retain their
declared camera coordinate convention; signed body angles retain the producer's
anatomical-forward/anatomical-left convention.

Global validation must check foreign-key closure and primary-key uniqueness
across all parts, not only inside each recording shard.

## Reader and consumer boundary

Add a validated storage-facing API with the conceptual shape:

```python
dataset = ValidatedBehaviorExportDataset.open(
    export_root,
    export_run_id,
    validate=True,
)

table = dataset.table("canonical_swim_bouts")
lazy = table.scan(
    columns=("recording_id", "bout_id", "duration_s"),
    predicate=...,
)
frame = lazy.limit(max_rows).collect()
```

The dataset/table handles must expose:

- the validated export, membership, and bundle-set manifests;
- exact manifest-selected part paths;
- table schema, grain, primary/foreign keys, units, provider dimensions, and
  capability policy;
- a true Polars `LazyFrame` with projection and predicate pushdown;
- optional Arrow scanner or bounded batch iteration;
- explicit bounded conversion for render payloads; and
- no globbing, source discovery, `latest` lookup, or direct Zarr fallback.

Each bounded query result must retain export/manifest/table identity, selected
columns and filters, aggregation grain, experimental-unit and weighting policy,
finite/missing/excluded counts, and any bound statistics-run identity. Cache
keys include the export-manifest digest, not only root, run, and table names.

Pandas is an optional bounded edge adapter, not an internal dependency. Marimo,
static plotting, and statistics consume normalized scientific/query payloads
above this reader. Renderers do not become storage or scientific authorities.

The current group explorer may continue to render existing tables while this
API is introduced. New generic Core Behavior and successor panels should use
the shared lazy reader instead of eagerly collecting whole tables into Python
dictionaries.

If a later report includes static artifacts, the report manifest binds this
exact export-manifest digest and labels whether each artifact was rendered from
Parquet or copied from a recording Zarr. Co-location does not change an
artifact's source backend.

The existing capability resolver's table/column boolean is not sufficient for
this dataset because it discards upstream typed reasons. The new reader exposes
the complete `recording_capabilities` relation and may derive a cohort summary
from it, but cannot replace it with a union-of-columns capability list.

### Statistical boundary

The first export is a descriptive, selector-ineligible query dataset. It does
not publish inferential results or claim a confirmatory statistical model.
Frame, bout, chaser, event, threshold, and bin rows are repeated measurements.
A later statistics run must define a MetricSpec, recording-level aggregation,
weighting, missingness, clustering/resampling unit, multiplicity family, and
exploratory/confirmatory status, and must bind the exact source export manifest.
Any analysis whose predeclared model requires acquisition-batch adjustment is
blocked while batch identity is missing; batch must not be imputed.

## Execution DAG

```text
validate source task and identity incident
  -> build and validate durable 84-member membership manifest
  -> plan exact 80-member projection-receipt roster
  -> materialize/validate recording bundles as an LSF array
  -> serialize and validate the 84-member bundle-set manifest
  -> plan per-recording, per-table export shards
  -> run shard array with bounded max-active concurrency
  -> all-required-success barrier
  -> serialize shard inventory
  -> deterministic table fan-in / part adoption
  -> global schema, key, capability, source-closure, and inventory validation
  -> atomic immutable publication; commit manifest last
  -> optional selector-ineligible catalog projection
  -> Marimo/static/statistics canaries
```

Reuse the established LSF DAG and artifact patterns. Every job must record the
exact full Palette commit and absolute commit-pinned deployment path produced by
`scripts/deploy_palette_cluster_worktree.sh`. Concurrent work must not move the
shared `/groups` checkout.

The first fan-in keeps one independently validated Parquet part per recording
and table. It assembles those parts into the hidden publication generation and
commits their closed inventory; it does not concatenate every cohort row in one
parent process. Any later file-compaction pass is a separate immutable
generation with its own measured policy and numerical/inventory validation.

Recording workers perform no registry writes. A serialized finalizer may write
an explicitly derived, selector-ineligible catalog entry only after the full
publication and validation receipt exist. The first cohort run should omit
registry mutation entirely unless a separate indexing contract has been
implemented and reviewed.

### Resume and retry semantics

- The plan ID is a canonical digest of membership, bundle set, table contracts,
  parameters, and software commit.
- Shard output paths include the plan ID and member ordinal or recording ID.
- A retry reuses a shard only after validating its receipt, exact inputs, part
  inventory, and all part hashes.
- A missing, partial, mismatched, or unlisted output is recomputed in a new
  staging path; it is never repaired in place.
- A failed recording cannot be hidden by a successful fan-in of the remaining
  parts. Optional-capability omissions are allowed only through their declared
  policy and exact contributing-member roster.
- The final publication generation is immutable. An identical retry may return
  the already validated generation; any differing input or output under the
  same identity fails closed.
- Recording bundles currently include `created_at_utc` in their record digest.
  Deterministic retry therefore means validating and reusing the exact existing
  bundle bytes through `ensure_validated_recording_behavior_bundle`, not
  rebuilding a byte-identical file at a later timestamp.

## Validation and acceptance gates

### Membership and bundle-set validation

- [ ] Validate the source task schema, file digest, internal task digest, and
      all 84 source entries.
- [ ] Build a durable roster that does not require the `/tmp` registry snapshot
      to remain present.
- [ ] Bind and validate the subject-identity incident decision.
- [ ] Prove 84 unique recording IDs and 84 unique analysis-unit IDs.
- [ ] Prove exactly 80 complete projection receipts and four explicit invalid
      semantic-selection dispositions.
- [ ] Regenerate the original canary against the coherent `c0ec...` receipt.
- [ ] Materialize and validate exactly one bundle for every admitted member.
- [ ] Require one bundle-set record for every parent member, including the four
      members without bundles.
- [ ] Reject any recording/Zarr/receipt mismatch, selector path, duplicate,
      omission, or extra member.
- [ ] Preserve all capability states and reasons exactly.

### Shard and table validation

- [ ] Validate exact source-child receipts before reading payloads.
- [ ] For every published immutable source child, validate direct/consolidated
      metadata equivalence or the child's exact receipt-bound published
      metadata-generation proof; stale consolidated metadata fails.
- [ ] Require exact Arrow schema equality and embedded table-contract metadata.
- [ ] Check non-null primary-key components and global uniqueness.
- [ ] Check every foreign key against its exact parent table.
- [ ] Verify row-count conservation against persisted source array counts for
      each grain.
- [ ] Verify provider, track, coordinate, semantic-window, chaser, trial, bout,
      event, and bin identity.
- [ ] Verify units, finite/valid masks, missingness, censoring, and denominator
      fields.
- [ ] Verify histogram/bin count conservation where the successor contract
      declares it.
- [ ] Verify each part path, size, row count, SHA-256, Arrow schema digest, and
      membership in the closed inventory.
- [ ] Reject missing, foreign, duplicate, or unlisted parts.
- [ ] Reject required tables that are unexpectedly empty.
- [ ] Verify deterministic output on an identical retry.

### Publication and consumer validation

- [ ] Stage under a unique hidden generation and commit the manifest last under
      the existing publication lock/CAS rules.
- [ ] Prove an interrupted or failed publication leaves the prior selected
      generation unchanged.
- [ ] Re-open the committed generation exclusively through its manifest.
- [ ] Confirm Polars lazy scans preserve projection and predicate pushdown.
- [ ] Confirm Arrow batch iteration is bounded.
- [ ] Confirm reader/cache identity changes when the export-manifest digest
      changes, even when root/run/table names are unchanged.
- [ ] Confirm the export reader rejects direct-Zarr fallback and any part path
      not selected by the manifest.
- [ ] Compare a canary table numerically against the exact Zarr arrays or
      persisted summaries, including NaNs and validity states.
- [ ] Confirm Marimo and a static renderer produce equivalent normalized data
      for one shared view.
- [ ] Confirm statistics aggregate first to `recording_id` and never treat
      frame/bout/event/bin rows as independent analysis units.
- [ ] Confirm missing acquisition batch remains explicit and prevents any
      analysis that requires batch clustering from being labeled confirmatory.
- [ ] Keep selector eligibility, production authority, selector activation, and
      registry mutation false.

## Adversarial tests

At minimum, focused tests must reject:

- a changed source task with the old task digest;
- 83 or 85 membership rows;
- a duplicated recording, dataset, ordinal, or Zarr binding;
- an unrecorded exclusion or a fifth silently omitted invalid recording;
- use of the reused source subject UUID as `analysis_unit_id`;
- an inferred acquisition-batch value;
- a bundle bound to the other canary receipt generation;
- a re-digested bundle after child-source substitution;
- mixed incompatible method/schema/provider/coordinate/track identities;
- provider rows whose primary key omits provider identity;
- trial-gap rows represented as active trial membership;
- a bout fact duplicated into one row per chaser instead of placed in the
  association table;
- reconstructed individual visits or aligned samples absent from the source;
- changed bin edges or denominators;
- duplicate primary keys split across different Parquet parts;
- a part that exists on disk but is absent from the manifest;
- a manifest part whose bytes or Arrow metadata changed;
- a capability declared available because another recording has its columns;
- a partial shard reused after a failed attempt;
- source, parameter, contract, or commit drift under a retry;
- publication over an existing immutable generation;
- a reader that globs directories or follows `latest`;
- any call from the validated exporter to old `goodcopbadcop_*`,
  `_SOURCE_TABLE_BY_V2`, `_latest_run`, or legacy chaser-distance source
  adapters; and
- a new validated-behavior table opened under the existing generic v3 table
  contract, or an old v3 table opened under the new contract.

## Implementation edit map

The exact file layout may be refined during implementation, but responsibilities
should remain separated:

| Responsibility | Proposed location |
|---|---|
| Membership and bundle-set schemas/builders/validators | `src/fisheye/analysis_workflows/validated_behavior_cohort.py` |
| Historical/future roster and recording-bundle adapters | `src/fisheye/analysis_workflows/validated_behavior_cohort_adapters.py` |
| Bundle fanout planner and CLI | `src/fisheye/utils/materialize_validated_behavior_bundle_cohort.py` |
| New table contracts and Arrow schemas | `src/fisheye/analytics_exports/contracts.py`, `src/fisheye/analytics_exports/arrow_contracts.py` |
| Bundle-backed source-to-table adapters | `src/fisheye/analytics_exports/validated_behavior_adapters.py` |
| Shard and cohort export manifests/validators | `src/fisheye/analytics_exports/validated_behavior_cohort.py` |
| Atomic publication reuse | `src/fisheye/analytics_exports/publication.py` |
| LSF plan/submission frontend | `src/fisheye/cluster/` plus a thin tracked script in `scripts/` |
| Lazy manifest-selected reader | `src/fisheye/analytics_exports/query.py` or a dedicated `dataset.py` |
| Marimo/query integration | `src/fisheye/group_analytics_viewer/` and `apps/marimo/components/` |
| Membership/bundle unit tests | `tests/unit/fisheye/test_validated_behavior_cohort.py` |
| Adapter/schema/publication tests | `tests/unit/fisheye/test_validated_behavior_cohort_export.py` |
| Lazy consumer tests | `tests/unit/fisheye/test_validated_behavior_export_dataset.py` |
| LSF plan tests | `tests/unit/fisheye/test_plan_validated_behavior_cohort_export_bsub.py` |

The existing `export_cross_recording_analytics.py` remains a useful source of
contract, Arrow, part-inventory, and publication mechanisms. The first
implementation should extract or call reusable modules rather than add another
large protocol-specific mode to that monolith. Legacy source discovery and old
chaser-distance adapters must not become fallback paths for this exact export.

`export_provider_epoch_behavior_cohort.py` remains a selector-ineligible
talk-support diagnostic. Its custom cohort manifest, nullable identity,
consolidated-only source check, and local table contracts do not satisfy this
bundle-backed cohort boundary. Reuse its bounded table-writing ideas only after
the new strict bundle/source handle has validated the source.

The current generic `validate_export_run` rejects unknown table families and
requires the generic `palette.analytics_export` envelope. Reuse or extract its
inventory, Arrow-footer, row-count, and publication checks behind a dedicated
validated-behavior contract map; do not weaken its schema gate or pretend the
new envelope is generic v3.

Every new table has its own explicit contract ID and version under the
validated-behavior family. Readers must reject attempts to open these tables as
the existing `palette.analytics_export` v3 contracts, and vice versa. A
compatibly named legacy table never authorizes a legacy source adapter.

## Staged implementation checklist

### Phase 0 — Contract freeze

- [ ] Freeze schema IDs, versions, state/reason vocabularies, and canonical JSON
      digest rules.
- [ ] Freeze the recording-scoped identity and missing-batch policies.
- [ ] Freeze table names, grains, keys, provider dimensions, units, and
      capability policies for the compact first release.
- [ ] Decide and record allowed zero-row semantics per table.
- [ ] Unify the two currently divergent recording-analysis-unit policy IDs
      before reusing the provider-epoch plotting receipt.
- [ ] Freeze the exact selected track and bout signal/measurement-level policy
      for every admitted recording.
- [ ] Add the contracts and negative unit tests before live publication code.

### Phase 1 — Durable membership and bundle set

- [ ] Implement the one-time schema-v5 historical task importer.
- [ ] Implement the future frozen-cohort-v2 importer into the same membership
      interface.
- [ ] Validate the 84-member GoodBatBadBat manifest and four invalid states.
- [ ] Add deterministic per-recording bundle plans using the coherent receipt
      generation.
- [ ] Run an in-memory/read-only plan for all 84 members.
- [ ] Materialize and validate a new canary bundle.
- [ ] After required CI, fan out the other 79 bundles.
- [ ] Build and validate the 84-member bundle-set manifest.

### Phase 2 — Compact export canary

- [ ] Implement identity, capability, source-binding, semantic-epoch,
      controller-trial, canonical-bout, association, escape-event,
      epoch-behavior, radial, spatial, and alignment adapters.
- [ ] Emit one recording's independently validated Parquet parts and shard
      receipt.
- [ ] Compare every exported field and row count to its exact child.
- [ ] Test interrupted writes, tampering, mismatched retry, and manifest-last
      publication.
- [ ] Keep the canary selector-ineligible and outside the registry.

### Phase 3 — Compact 80-recording cohort export

- [ ] Deploy one clean commit-pinned cluster worktree after required CI is
      green.
- [ ] Freeze the exact export plan and table roster.
- [ ] Submit per-recording shards with bounded `max_active` concurrency.
- [ ] Require the all-success barrier for required tables.
- [ ] Serialize shard inventory and deterministic fan-in.
- [ ] Run global key, foreign-key, capability, source-closure, row-count, Arrow,
      and inventory validation.
- [ ] Publish the immutable generation and validation receipt.
- [ ] Preserve 84 membership/capability rows and exact contributing-member
      rosters for every scientific table.

### Phase 4 — Lazy consumers and first cohort figures

- [ ] Implement the manifest-selected dataset/table handles.
- [ ] Migrate one Core Behavior cohort view and one chaser cohort view.
- [ ] Enforce bounded collection at the renderer boundary.
- [ ] Add provider and capability filters without source rediscovery.
- [ ] Validate numerical parity with the recording-local source adapters.
- [ ] Record bundle/export/table/filter/aggregation/display provenance in every
      rendered artifact.

### Phase 5 — Optional dense projections

- [ ] Measure row counts, bytes, row-group sizes, filter latency, memory, and
      publication time on a one-recording sample.
- [ ] Implement streaming `provider_motion_samples` shards.
- [ ] Implement provider-explicit `chaser_relative_samples` shards.
- [ ] Add body-alignment and exact active-trial-membership samples only when a
      consumer requires them.
- [ ] Keep every worker's file ownership disjoint; if any Zarr staging is ever
      introduced, require whole non-overlapping physical chunk ownership.
- [ ] Publish dense tables as optional manifest-bound extensions, not as a
      prerequisite for the compact release.

### Phase 6 — Later scientific successors

- [ ] Add individual near-field visits before ring-entry artifacts.
- [ ] Add exact video/frame mapping before ring-entry media.
- [ ] Add persisted event-aligned escape samples before trajectory tables.
- [ ] Add response-regimes v2 before regime exports.
- [ ] Add reviewed gaze and chaser-tail successors only after their upstream
      authorities are accepted and bundle-bound.
- [ ] Add each capability through its own schema, receipt, bundle state, export
      table contract, and focused validation.

## Required CI and deployment boundary

Documentation alone authorizes no execution. Each implementation branch must
pass every required CI check before merge, integration, shared-checkout update,
production use, or a merge-ready claim. Failed, cancelled, timed-out, or
accidentally skipped required checks remain blocking.

Experimental evidence may be collected from a detached, commit-pinned
deployment after the relevant focused tests pass, but the deployment remains
selector-ineligible and non-production until required CI is green. Every LSF
plan and shard receipt must record the exact `PALETTE_GROUPS_REPO` path and full
commit. Do not repoint an existing plan to a different deployment.

## Deferred optimization after correctness

The existing generic exporter accumulates rows in a parent process. The new
pipeline should write validated per-recording parts directly and adopt them by
manifest, which provides both bounded memory and resumability. Further
optimization should be driven by recorded measurements:

- rows and bytes per table and recording;
- source-read and Parquet-write bytes;
- extraction, validation, and publication wall time;
- peak RSS and worker CPU utilization;
- Parquet file and row-group counts;
- predicate/projection pushdown latency; and
- shard reuse versus recomputation counts.

Receipt-aware reuse may remove repeated deep validation only when the receipt
is cryptographically bound to the exact immutable source, contract, parameters,
and expected output. Optimization cannot weaken current-source closure,
manifest inventory, primary-key, or numerical parity checks.

## Definition of done

The first merged GoodBatBadBat dataset is complete when:

- one durable manifest accounts for all 84 parent recordings and the identity
  incident;
- exactly 80 complete recording bundles are bound, with four explicit invalid
  semantic-selection dispositions;
- compact normalized scientific tables are built only from bundle-complete
  capabilities;
- every table declares its grain, keys, provider dimensions, units,
  contributing-member roster, capability policy, and exact source lineage;
- every shard and part is receipt-backed and globally validated;
- one immutable manifest-selected publication is committed atomically;
- a lazy reader can serve bounded Marimo, static, and statistical consumers;
- numerical parity is proven against recording-local authorities;
- statistical consumers use recording-level experimental units and expose the
  84-parent/80-admitted denominator distinction;
- unavailable future products remain explicitly unavailable rather than being
  reconstructed during export;
- no recording Zarr, source receipt, selector, active registry authority, or
  shared checkout was mutated; and
- all required CI is green before the implementation is merged or described as
  operational.

## Implementation addendum — 2026-08-31 generic contract slice

The first implementation slice enforces the generic composition boundary
above:

- `validated_behavior_cohort.py` implements protocol-independent membership,
  bundle-set, capability-contract, deterministic digest, root-confinement,
  analysis-unit, batch-missingness, and safety validation. Capability names
  are supplied by a profile and are not compiled into the core.
- `validated_behavior_cohort_adapters.py` contains the replaceable adapters
  for historical composable-chaser task v5, future frozen-cohort v2, exact
  projection receipts, and validated-recording-behavior bundle v1.
- `materialize_validated_behavior_bundle_cohort.py` provides read-only planning
  and validation commands. It writes external JSON contracts only and has no
  Zarr, registry, selector, or production mutation path.
- The historical adapter requires an explicit self-digested invalid-member
  document. Every remaining parent must carry one exact receipt from the
  requested generation; missing files fail and never alter membership.
- The historical temporal-proxy caveat and incident-authorized recording unit
  are adapter-bound policy data. GoodBatBadBat member IDs, counts, and protocol
  hashes remain input records, not implementation constants.

Focused validation currently covers arbitrary non-chaser capability names,
duplicate and substituted identity, source-subject misuse, exact parent-roster
closure, both membership import profiles, explicit invalid decisions, bounded
CLI output, and recording-bundle capability preservation. A live read-only
planning pass over the current source task proved 84 parent members, 80 exact
`c0ec5dc0...` admissions, four explicit invalid members, and 80 admission
receipt bindings. No durable membership was written during that pass because
the implementation did not yet have a committed software identity.
