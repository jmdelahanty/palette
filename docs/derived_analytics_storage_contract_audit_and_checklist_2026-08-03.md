# Derived Analytics Storage Contract Audit And Implementation Checklist

Date: 2026-08-03

Status: active implementation checklist; reconciled through the reviewed
eye-angle and track-kinematics benchmark/catalog checkpoints `9c771842`,
`c56b3103`, and `bf23020c`, plus the stimulus-step Arrow checkpoints
`f53c0d7a` and `93e81b12`; correctness, executable
catalog, shared planner, serialized registry, compact rematerialization, and
benchmark foundations are integrated. All thirteen current catalog families
have explicit unpromoted physical candidates. No production profile,
scientific selector, registry authority, or canonical data changed by these
checkpoints.

Benchmark coverage is now separately executable rather than inferred from
candidate presence. Ten of thirteen families have an executable
source/candidate read matrix; none yet has the complete writer, publication, physical-I/O,
representative-scale, and real-consumer evidence required for promotion. The
catalog binds any future measured/executed claims to an immutable evidence
receipt and versioned passing gate; complete catalog coverage still does not
authorize profile activation. The initial policy conservatively requires a
Crimson gate for every current family; that requirement may be narrowed only
by a later explicit consumer census, not by the absence of an adapter today.

## Purpose

This document records the current contract coverage for Palette analytics
outputs and defines the implementation sequence for closing the remaining
logical-schema, physical-storage, publication, selection, and consumer gaps.

It follows the detection, crop, keypoint, and subject-mask contract work. Those
upstream observation families are not re-audited here, except for the
`body_frame_runs` and `keypoint_quality_runs` companion analytics that already
demonstrate the newer shared schema and byte-planner design.

The audit used the `sun` checkout at commit `83ac49be`. Implementation proceeds
on the isolated coordination branch named below; it has made repository changes
but no production-data, registry-authority, profile-promotion, or scientific
selector changes.

## Executive Decision

Palette does **not** yet have uniform production-grade storage contracts for
all analytics outputs.

Thirteen maintained array-bearing derived families now have executable catalog
entries. Seven production authorities use the shared atomic materializer
boundary:

- track kinematics;
- swim bouts;
- bout kinematics;
- eye angles;
- subject shape;
- tail kinematics; and
- stimulus response.

Tail-posture views, bout classification, stimulus epochs, detection occupancy,
session occupancy, and chaser distance are also cataloged, truthfully, as
guarded direct production writers rather than falsely claiming
atomic-materializer ownership. None of the thirteen uses the shared
byte-budgeted planner as its production default yet. All thirteen expose
explicit selector-ineligible candidates; eleven publish through the shared
atomic boundary, while tail-posture and bout-classification retain guarded
direct candidate writers. All thirteen have serialized registry projection for
eligible authorities. The ten embedded chaser components remain independently
sealed component surfaces rather than separate top-level run families.
Several legacy or in-place writers remain explicitly classified compatibility
or maintenance surfaces.

The target is not one scientific schema for every datatype. The target is one
contract system in which every persisted output is classified and every
scientific array has:

1. an exact logical identity;
2. an exact persisted representation;
3. an access- and byte-aware physical plan;
4. a safe publication and selection lifecycle; and
5. a real consumer and benchmark workload.

## Definition Of A Complete Analytics Storage Contract

A maintained analytics family is complete only when all of the following are
true.

### Logical schema

- Every array path is bound to a versioned logical contract.
- Dtype, rank, named axes, shape constraints, units, coordinate domain, and
  fill/null/sentinel semantics are exact.
- Required, optional, derived-cache, compatibility, and forbidden arrays are
  explicit.
- Row identity and source lineage are exact and fail closed.
- Unexpected arrays or attributes cannot silently become scientific authority.

### Physical storage

- Each array declares an access class and write mode.
- Inner chunks and outer shards are derived from uncompressed bytes and access
  shape rather than a shared row-count literal.
- The Zarr version, codec chain, compression level, checksum, shard-index
  configuration, and object estimates are manifest-bound.
- Parallel writers own complete, non-overlapping physical chunks or shards.

### Publication and selection

- Compute occurs in node-local scratch when appropriate.
- Publication is immutable, validated, consolidated, and atomic.
- Completion, selector eligibility, authority selection, and registry
  projection are separate fail-closed steps.
- A completed but selector-ineligible candidate cannot be reused implicitly.
- Copy verification is strong enough for the scientific risk of the family.

### Consumer and performance evidence

- Palette and Crimson readers select one exact schema version without dtype or
  path probing for current data.
- Legacy fallbacks require an explicit compatibility mode.
- Each writer has a deterministic benchmark covering its real access patterns.
- Logical equality, publication cost, object count, transfer, latency, and peak
  RSS are recorded before a physical profile is promoted.

## Current Coverage

| Surface | Current contract position | Main remaining work |
| --- | --- | --- |
| Track kinematics | Exact 69-array per-track core, closed 35-array physical bundle, run identities, legacy exclusions, materializer, and registry contract; the v2 candidate flattens the two structured lineage records into five exact primitive arrays while retaining float64 position authority. A hardened fresh-process v1-source/v2-candidate matrix exercises the complete no-physical logical surface through the public v1 reader and an explicit diagnostic-only v2 adapter, with live workload replay and hard nonpromotion | Run representative short/full matrices with physical-I/O tracing and real consumers; benchmark the optional 35-array physical bundle separately before any writer/profile promotion |
| Subject shape | Exact v4 full-anatomy component/relation/row-index inventory, producer-sealed source binding, lineage, atomic publication, strict reload, serialized registry projection, an explicit selector-ineligible byte-planned candidate, and a hardened fresh-process source/candidate read matrix preserving every dtype, fill, decoded payload, and installed transform | Run representative publication, physical-I/O, Palette-consumer, and Crimson-consumer gates before any profile promotion |
| Tail kinematics | Exact 21-array core plus an all-or-none two-array revision bundle, coordinate/lineage semantics, atomic publication, registry projection, and an explicit selector-ineligible byte-planned candidate | Run full-duration producer/reader benchmarks before profile promotion |
| Eye angles | Exact 41-array compact-v7 schema, deep manifests, strict maintained readers, registry projection, and an atomic selector-ineligible byte-planned candidate with exact failure-visibility repair. Its hardened fresh-process source/candidate read matrix now binds all logical-input roles to exact nested authorities, validates all 41 arrays and physical declarations, and leaves the private candidate adapter explicitly non-authoritative | Run the five-repetition full-duration matrix, add physical-I/O tracing if required, and gate real Palette/Crimson consumers before profile promotion |
| Swim bouts | Exact compact-v8 whole-run array manifest, authoritative selection, serialized registry projection, an immutable selector-ineligible byte-planned candidate publisher, and the first shared fresh-process candidate runner with live track-coordinate and frame-axis resolution | Run representative short/full publication and real-consumer gates with external physical-I/O tracing before profile promotion |
| Bout kinematics | Exact compact-v7 manifest, authoritative selection, serialized registry projection, an immutable selector-ineligible byte-planned candidate publisher, and a shared fresh-process candidate runner that resolves both its immutable swim-bout source and their common live track-coordinate authority | Run representative short/full publication and real-consumer gates with external physical-I/O tracing before profile promotion |
| Stimulus response | Exact opt-in compact-v3 schema and selector-ineligible shared-planner/factory candidate with closed bundles, semantic fills, pinned codecs, immutable atomic publication, and direct/consolidated validation; legacy v2 remains the default compatibility path | Benchmark the v3 candidate through real producers/consumers before any default change |
| Body frame | Exact ten-array schema, byte planner, strict manifest, and consolidated publication | Move from selector-ineligible companion evidence to an explicitly governed production lifecycle when authorized |
| Keypoint quality | Exact diagnostic schema, byte planner, strict manifest, and consolidated publication | Add explicit production lifecycle/registry policy when authorized |
| Derived keypoint metrics | Current v2 keeps profile-specific diagnostics in immutable source-bound `keypoint_quality_runs`; refined v2 retains compact acceptance gates and forbids legacy triangle arrays | Add any future skeleton-specific triangle metrics as a versioned quality profile, not universal refined-keypoint arrays |
| Tail-posture view | Exact typed arrays, guarded lifecycle, and an explicit selector-ineligible byte-planned candidate with semantic fills | Benchmark the candidate and decide whether to migrate to the atomic materializer |
| Bout classification | Exact compact-v2 manifest, guarded activation, serialized registry projection, and an explicit selector-ineligible byte-planned candidate with semantic fills | Benchmark the candidate through its real consumer |
| Track visualization | Shared PNG/spec byte artifact contract | Keep classified as an artifact rather than a numeric scientific run |
| Stimulus epochs | Current v1 direct authority is centrally cataloged; an exact 12-array v2, candidate-owned lineage/manifest, shared byte plan, atomic publication, failure repair, and a strict named-run v2 consumer are implemented without changing selection | Run real source/candidate archive benchmarks before any writer/default promotion |
| Detection/session occupancy | Separate epoch-aligned and full-session authorities have closed 30-array and 29-array manifests, exact dtypes/axes/units/roles, central stage/catalog/registry ownership, and selector-ineligible shared-planner rematerialization | Benchmark both families before any production-profile promotion |
| Chaser-distance base | Central current v1 logical/production contract plus an exact 30-array sealed-base v2 physical candidate, source-authority binding, byte-planned rematerialization, atomic selector-ineligible publication, decoded hashes, and persisted direct/consolidated metadata equivalence | Run representative short/full writer, publication, and consumer benchmarks before any profile promotion |
| Chaser components | Protocol-neutral schemas, payload-bound manifests, all ten maintained writers routed through node-local sealed immutable publication, exact runner receipts, and exact self-digested handles propagated through maintained chained consumers and batch orchestration | Benchmark the component consumers and complete consolidated recovery coverage; keep them embedded rather than inventing top-level run families |
| Cross-recording Parquet exports | Versioned table/manifest contracts with immutable, manifest-exclusive atomic generations; a closed digest-bound Arrow envelope now freezes nine unique exact schemas across writing, staging, and manifest-selected reads: position occupancy, recording summary, stimulus steps, per-fish stimulus-step summary, the three baseline tables, and both group-statistics tables | Freeze the canonical envelope's remaining 21 inferred tables, baseline-strategy's four inferred outputs, and training-response's three inferred outputs only after producer semantics and nullability are resolved |
| Core workflow exports | Declared planning nodes for sampled kinematics, summaries, eye traces, and tail traces | Implement adapters and exact export contracts |
| Legacy/in-place outputs | `speed_runs`, swim-bout statistics, and stimulus/chaser mutation paths remain | Classify as legacy/maintenance or migrate; do not leave them implicitly current |

## Evidence And Known Disagreements

The executable catalog explicitly says physical policy remains stage-owned and
exposes `byte_planner_adopted` as the migration boundary
(`src/fisheye/analysis_workflows/storage_contract_catalog.py`). Its thirteen
entries include the current chaser-distance v1 authority while the separate
candidate catalog records its unpromoted 30-array sealed-base projection.
`src/fisheye/analysis_workflows/storage_benchmark_catalog.py` independently
records benchmark-adapter and execution evidence, preventing a candidate from
being mistaken for measured or promotable storage.

The storage catalog now includes tail-posture view and bout classification with
their real guarded-direct-writer owners
(`src/fisheye/analysis_workflows/storage_contract_catalog.py:294-335`). Their
canonical parent mappings live in the shared run-parent catalog
(`src/fisheye/shared/stage_run_groups.py:41-42`) rather than availability-local
patches. Track-kinematics visualization remains intentionally classified as a
byte artifact rather than an array-bearing scientific run.

The shared columnar helper is an earlier-generation physical policy: it uses
4,096-row one-dimensional chunks, 1,024-row multidimensional chunks, and
262,144-row requested shards
(`src/fisheye/shared/zarr/columnar.py:31-94`). It passes chunks and shards to
array creation but does not freeze an explicit codec chain there
(`src/fisheye/shared/zarr/columnar.py:141-155`).

Production compact-v2 stimulus-response fields and dtypes are still inferred
from the current row payload (`src/fisheye/analysis/stimulus_response.py`), but
the opt-in compact-v3 path now has an exact closed schema, strict coercion, and
all-or-none optional bundles. The older `analysis_stage_arrays` declarations
remain compatibility/input declarations rather than the compact-v3 persisted
authority.

The older canonical analytics matrix documents seven maintained families
(`docs/analytics_storage_schema_matrix.md:31-49`). It now trails the executable
thirteen-family catalog and remains incomplete as a full writer census. The earlier
reconciliation
correctly leaves the per-array inventory, byte-planner migration, and consumer
benchmarks open
(`docs/derived_analytics_storage_reconciliation_2026-08-01.md:102-124`).

Some chaser documentation and visualization labels still say “refined
detection.” The current canonical implementation instead accepts exactly
`detect_runs/<run>` and explicitly rejects refined or inferred paths
(`src/fisheye/analysis/chaser_distance_runs.py:790-804` and
`src/fisheye/analysis/chaser_distance_coordinate_publication.py:343-357`). The
stage dependency therefore records `detect`, not `refined_detect`; code is the
authority until a future version adds a strict refined-observation input.

## Correctness Blockers Before Physical Optimization

The findings in this section are the audit baseline. Phases 1 and 2 above have
closed the discovery, reader-fallback, and Parquet-generation blockers. The
chaser component blocker remains open for Phase 5.

### Authoritative run discovery

The generic workflow availability path follows completion pointers but does not
require `stage_selector_eligible`. The atomic publisher records completion and
pointers before final selector activation. An interrupted activation can
therefore leave a completed, ineligible candidate discoverable for reuse.

Evidence:

- `src/fisheye/analysis_workflows/materializers/atomic_run_publisher.py:873-907`
- `src/fisheye/analysis_workflows/availability.py:332-401`
- `src/fisheye/utils/execute_analysis_workflow.py:221`
- canonical fail-closed resolver precedent:
  `src/fisheye/shared/zarr_run_completion.py:394`

### Maintained-reader fallback behavior

Swim-bout discovery can use lexical fallback without proving completion or
eligibility, and some eye-angle consumers read `latest` directly.

Evidence:

- `src/fisheye/analysis/swim_bout_io.py:158,954`
- `src/fisheye/analysis/eye_angle_io.py:470,587`
- `src/fisheye/visualization/visualize_eye_angle_overlays.py:707`

### Cross-recording export generations

Parquet parts are written directly into the final generation directory. An
overwrite does not first remove or isolate older parts. Validation checks the
manifest-listed set, while at least one consumer globs every Parquet part. A
rerun with fewer parts can therefore validate one set and read another.

Evidence:

- `src/fisheye/utils/export_cross_recording_analytics.py:5382-5435,5606-5613`
- `src/fisheye/analytics_exports/validation.py:87-178`
- `src/fisheye/utils/plot_cross_recording_bout_kinematics.py:39-51`

### Chaser component dependency selection

All ten maintained component writers now publish independently sealed,
selector-ineligible immutable candidates, and writer plus runner receipts carry
an explicit validated component handle. Bout response now consumes an exact
egocentric-bearing handle, and escape events consumes an exact bout-response
handle; both persist the upstream component-manifest digest in their lineage.
The remaining correctness boundary is the rest of the chained consumers and
exports. Candidates remain intentionally undiscoverable until each such
consumer binds an exact component manifest digest or a separately reviewed
selector activation.

Evidence:

- `src/fisheye/analysis/chaser_component_writer.py`
- `src/fisheye/analysis_workflows/materializers/chaser_component.py`
- `docs/chaser_component_publication_contract_v1.md`

## Implementation Checklist

### Parallel Worktree Execution Plan

Parallel implementation is allowed only when each lane owns a disjoint set of
production modules and tests.  Shared catalogs, shared storage-policy types,
publication infrastructure, registry projection, and this coordination
checklist remain integration-lane surfaces.  A lane must stop and return for
review instead of editing another lane's files.

Historical lanes are pinned from the coordination history and do not modify
production archives, selectors, or registries. New implementation lanes must
record and start from the current clean coordination `HEAD`, not from one of
the older worktree heads listed by `git worktree list`. A historical dirty
worktree is evidence to reconcile, not a safe base for new work.

| Lane | Worktree / branch | Base commit | Owned implementation surface | Integration rule |
| --- | --- | --- | --- | --- |
| Coordination | repository root / `agent/palette/derived-analytics-storage-contracts-20260803` | through `bf23020c` plus this checklist reconciliation | Shared catalogs, checklist, cross-family tests, reviewed cherry-picks | Serial owner; never rebased onto an unreviewed lane |
| Surface classification | `/tmp/palette-analytics-surface-classification-20260803` / `agent/palette/analytics-surface-classification-20260803` | `4923757f` | Closed classification catalog and its focused tests | Integrated as `a2668075` and `f178e315`; it classifies rather than promotes |
| Eye-angle exact schema | `/tmp/palette-eye-v7-clean-20260803` / `agent/palette/eye-v7-clean-20260803` | `dd4c0f74` | Reusable analytics array declaration, exact 41-array eye-angle schema, deep writer/reader/readiness validation, and explicit v2-v6 compatibility boundaries | Reviewed as ready, committed as `3025d66e`, and integrated as `c1976300` |
| Eye-angle byte candidate | historical isolated lane | `28f0132c` through `ac6b8bc3` | Exact 41-array planner/factory candidate and semantic fills | Integrated; remains explicit and selector-ineligible |
| Tail-kinematics byte candidate | historical isolated lane | integrated as `600fab64` | Exact 21-array core, revision bundle, planner/factory candidate, metadata equivalence, and lifecycle guards | Integrated; remains explicit and selector-ineligible |
| Track-kinematics exact schema and candidate | historical isolated lanes plus coordination hardening | integrated through `f273be11` | Exact current motion vocabulary, primitive v2 lineage projection, source-derived inventory, byte-planned candidate, and run-parent binding | Integrated; remains selector-ineligible and unpromoted |
| Shared compact rematerialization | coordination lane | integrated as `5d7fb30c` | Exact declaration-bound replanning, frame-axis growth, whole-physical-unit writes, receipt and metadata validation | Infrastructure only; family adoption remains explicit and opt-in |
| Shared compact candidate publication | coordination lane | integrated from `d82dcf41` | Immutable selector-ineligible swim-bout and bout-kinematics candidates, logical hashes, local and authoritative-root metadata equivalence, and selector non-mutation | Candidate evidence only; benchmark and promotion remain pending |
| Chaser scientific writer adoption | historical isolated lane | integrated as `fc6a48c5` | All ten component writers, sealed staging capability, atomic component publisher, focused tests, and lifecycle doc | Integrated as candidate publication; explicit dependency handles and activation remain separate |
| Chaser dependency/runner receipts | coordination lane | integrated as `34c067e5` and `21e115b3` | Self-digested explicit handles, detached exact reads, writer receipt v2, and cluster target receipt v2 | Integrated; scientific chained consumers and selector activation remain separate |
| Chaser remaining exact consumers | `/tmp/palette-chaser-remaining-handles-20260803` / `agent/palette/chaser-remaining-handles-20260803` | integrated as `e179075d` | Exact egocentric/quadrant handles in gaze, near-field, batch orchestration, and explicit historical-wrapper compatibility | Reviewed and integrated; selector activation remains separate |
| Archive metadata hardening | coordination lane | integrated as `0a4985fe` | Fork-safe archive lock, serialized direct consolidators/activations, exact failed-tombstone repair, bout rollback consolidation, and subject-mask unknown-ack recovery | Independently reviewed ACCEPT; arbitrary legacy/external mutators still require quiescence until migrated |
| Eye-angle atomic candidate | `/tmp/palette-eye-angle-atomic-candidate-v2-20260803` / `agent/palette/eye-angle-atomic-candidate-v2-20260803` | integrated as `c658dfcc` and `10bf957e` | Direct byte-planned 41-array candidate, atomic non-promoting publication, exact consolidated metadata, containment guards, and failure repair | Reviewed and integrated; benchmark and promotion remain pending |
| Physical candidate catalog | coordination lane | integrated and extended through this checkpoint | Executable separation of thirteen unpromoted candidate profiles from production logical contracts, including exact run parent, owner, atomic/direct mode, consolidation, and repair state | Never infer production adoption or promotion from candidate membership |
| Stimulus-response v3 byte candidate | `/tmp/palette-stimulus-response-v3-candidate-20260803` / `agent/palette/stimulus-response-v3-candidate-20260803` | integrated as `40434306` | Exact opt-in candidate writer, storage receipt, metadata equivalence, immutable publisher, strict reader, and adversarial tests | Integrated; benchmark evidence and promotion remain pending |
| Exact-tabular read benchmark | `/tmp/palette-bout-storage-benchmark-20260803` / `agent/palette/bout-storage-benchmark-20260803` | integrated as `e348d53f` | Fresh-process deterministic source/candidate runner for swim bouts and bout kinematics, exact equality, access-class reads, full scans, storage/RSS/timing evidence, and strict output contracts | Integrated; authorized short/full-duration execution and physical-I/O tracing remain pending |
| Subject-shape byte candidate | `/tmp/palette-subject-shape-byte-candidate-20260803` / `agent/palette/subject-shape-byte-candidate-20260803` | integrated as `ce6fdfc9`, `b9da5d36`, and `039eccc7` | Producer-sealed full-v4 inventory, exact semantic fills, byte-planned rematerialization, atomic publication, and persisted metadata equivalence | Independently reviewed ACCEPT; benchmarks remain pending |
| Stimulus epoch and occupancy candidates | isolated family lanes plus coordination catalog work | integrated before `f273be11` | Exact 12/30/29-array contracts, byte-planned candidates, atomic rematerialization, and benchmark-harness coverage | Integrated; strict stimulus-v2 consumer and representative benchmarks remain pending |
| Chaser-distance sealed-base candidate | `/tmp/palette-chaser-distance-storage-candidate-20260803` / `agent/palette/chaser-distance-storage-candidate-20260803` | integrated as `563d7720` | Exact source-sealed 30-array projection, byte-aware plan, source/candidate hashes, persisted metadata equality, and atomic non-promoting publication | Independently reviewed ACCEPT; full-duration benchmark remains pending |
| Chaser-distance sealed-base read matrix | `/tmp/palette-chaser-distance-base-benchmark-20260803` / `agent/palette/chaser-distance-base-benchmark-20260803` | integrated as `14013b09` | Explicit source/candidate parents, authority/manifest/receipt bindings, rotated fresh processes, exact decoded equality, metadata and archive guards, CPU/wall/RSS/object/byte evidence, and hard-coded nonpromotion | Independently reviewed ACCEPT; representative execution, physical-I/O tracing, real consumers, and promotion remain pending |
| Stimulus-epoch v2 consumer | `/tmp/palette-stimulus-epoch-v2-consumer-20260803` / `agent/palette/stimulus-epoch-v2-consumer-20260803` | integrated as `72d59c84` | Explicit named selector-ineligible v2 read, complete direct/consolidated metadata gate, exact lifecycle/schema/lineage/manifest/receipt validation, eager backend-neutral rows, and typed explicit-only v1 compatibility | Independently reviewed ACCEPT; source/candidate benchmark and any selection change remain pending |
| Stimulus-epoch source/candidate read matrix | `/tmp/palette-stimulus-epoch-v2-read-benchmark-20260803` / `agent/palette/stimulus-epoch-v2-read-benchmark-20260803` | integrated as `5293d4dd` | Strict v1-source/v2-candidate reads, embedded executable lineage/manifest/storage documents, coordinated-rebind rejection, complete decoded equality, fresh processes, metadata/archive guards, and hard nonpromotion | Independently reviewed ACCEPT; representative execution, physical-I/O tracing, consumers, and promotion remain pending |
| Stimulus-response compact-v3 source/candidate read matrix | `/tmp/palette-stimulus-response-v3-read-benchmark-20260803` / `agent/palette/stimulus-response-v3-read-benchmark-20260803` | integrated as `a3814845` | Explicit compatibility-source/candidate names, executable v1/v2 schemas, offline-replanned HTTP-v1 receipt, complete decoded equality, rotated fresh processes, metadata/archive guards, and hard nonpromotion | Independently reviewed ACCEPT; representative execution, writer/publication timing, physical-I/O tracing, consumers, and promotion remain pending |
| First exact Arrow export schema | `/tmp/palette-analytics-arrow-dtype-contracts-20260803` / `agent/palette/analytics-arrow-dtype-contracts-20260803` | integrated as `379b9262` | Closed digest-bound Arrow envelope, exact 62-field position-occupancy schema, writer/staging/read validation, and an explicit census of 36 inferred compatibility schemas | Independently reviewed ACCEPT; remaining exact schemas, zero-row hardening, and production evidence remain pending |
| Second exact Arrow export schema | `/tmp/palette-recording-summary-arrow-contract-20260803` / `agent/palette/recording-summary-arrow-contract-20260803` | integrated as `9da4de70` | Exact ordered 32-field recording-summary schema, stable nullable capability columns, manifest-exact zero-row behavior, strict writer/footer/selected-reader validation, and registry rejection of inferred minimal summaries | Independently reviewed ACCEPT; at that checkpoint 35 enumerated schemas remained, and production consumer evidence was pending |
| Third exact Arrow export schema | `/tmp/palette-baseline-summary-arrow-contract-20260803` / `agent/palette/baseline-summary-arrow-contract-20260803` | integrated as `55cd2797` | Exact ordered 95-field baseline-behavior-summary schema, fixed eight-key source projection, nullable FPS discrepancy, strict writer/footer/selected-reader validation, and representation-only source test doubles | Independently reviewed ACCEPT; source authority remains quarantined; at that checkpoint 34 enumerated schemas and cross-language evidence remained |
| Fourth exact Arrow export schema | `/tmp/palette-baseline-time-bins-arrow-contract-20260803` / `agent/palette/baseline-time-bins-arrow-contract-20260803` | integrated as `cc75c755` | Exact ordered 77-field baseline-behavior-time-bin schema, fixed 38-key metric vocabulary, nullable FPS discrepancy, strict writer/footer/selected-reader validation, and representation-only source test doubles | Independently reviewed ACCEPT; source authority remains quarantined; at that checkpoint 33 enumerated schemas remained |
| Fifth exact Arrow export schema | `/tmp/palette-baseline-samples-arrow-contract-20260803` / `agent/palette/baseline-samples-arrow-contract-20260803` | integrated as `359f072b` | Exact ordered 71-field baseline-kinematic-samples schema, fixed 32-key sample vocabulary, explicit null-not-sentinel semantics, strict writer/footer/selected-reader validation, and representation-only full-resolution source test doubles | Independently reviewed ACCEPT; source authority and physical part-size benchmarking remain pending; at that checkpoint 32 enumerated schemas remained |
| Subject-shape v4 source/candidate read matrix | `/tmp/palette-subject-shape-v4-read-benchmark-20260803` / `agent/palette/subject-shape-v4-read-benchmark-20260803` | integrated as `fa37a16f` | Exact transformed/untransformed logical equality, executable candidate plan and metadata replay, distinct PID/role/order binding, componentwise path containment, rotated fresh processes, and hard nonpromotion | Independently re-reviewed ACCEPT after four blocker fixes; representative execution, physical-I/O tracing, real consumers, and promotion remain pending |
| Sixth exact Arrow export schema | `/tmp/palette-stimulus-steps-arrow-contract-20260803` / `agent/palette/stimulus-steps-arrow-contract-20260803` | integrated as `f53c0d7a` | Exact ordered 60-field stimulus-step schema, closed maintained moving/concentric child vocabularies, null/unit/interval semantics, strict writer/footer/selected-reader validation, and dynamic-column rejection | Independently reviewed ACCEPT; source selection authority and representative query evidence remain separate |
| Seventh exact Arrow export schema | `/tmp/palette-stimulus-step-summary-arrow-contract-20260803` / `agent/palette/stimulus-step-summary-arrow-contract-20260803` | integrated as `93e81b12` | Exact ordered 38-field per-fish stimulus-step summary, corrected `(recording_id, fish_id, step_index)` key, real two-fish coverage, optional bout bundle, and strict writer/footer/selected-reader validation | Independently reviewed ACCEPT; value-level global key scanning and representative query evidence remain separate |
| Eighth and ninth exact Arrow export schemas | `/tmp/palette-group-statistics-arrow-contract-20260803` / `agent/palette/group-statistics-arrow-contract-20260803` | integrated as `f7d02e9e` from reviewed checkpoint `3a4ac29b` | Exact ordered 45-field inferential and 30-field descriptive group-statistics schemas, closed metric-unit/effect-size/CI semantics, deterministic result identities, schema-bearing zero-row generations, strict viewer validation, and explicit read-only inferred-v2 compatibility | Independently reviewed ACCEPT; 158 tests passed with 5 expected compatibility xfails; production publication evidence remains separate |
| Eye-angle compact-v7 source/candidate read matrix | `/tmp/palette-eye-angle-v7-read-benchmark-20260803` / `agent/palette/eye-angle-v7-read-benchmark-20260803` | integrated and hardened through `35995e23`; cataloged by `9c771842` | Complete 41-array decoded equality, executable schema/storage replay, exact publication and nested materialization bindings, logical-input role identities, fresh processes, archive guards, and hard nonpromotion | Independently re-reviewed ACCEPT; full-duration execution, physical-I/O tracing, public candidate API decision, real consumers, and promotion remain pending |
| Track-kinematics v1/v2 source/candidate read matrix | `/tmp/palette-track-kinematics-v2-read-benchmark-20260803` / `agent/palette/track-kinematics-v2-read-benchmark-20260803` | integrated as `c56b3103`; cataloged by `bf23020c` | Complete no-physical logical projection, public v1 source reader, diagnostic-only v2 adapter, exact backend/verification policy, live selector/receipt/workload replay, fresh processes, and hard nonpromotion | Independently re-reviewed ACCEPT; representative execution, physical-I/O tracing, maintained v2 consumer decision, and the optional 35-array physical bundle remain pending |
| Shared candidate execution contract | coordination lane | integrated as `71d13675` | Closed 13-family typed adapter catalog; family-specific decoded-equality IDs; live family-suite validation; node-local and benchmark-namespace requests; collision-resistant protected-state snapshots; parent-observed fresh-process identity; exact coordinate evidence; complete-only 11-phase receipts; atomic acceptance/tombstoning; and hard selector/registry/profile/canonical nonmutation. Swim bouts and bout kinematics are the first two implemented adapters | Independently re-reviewed ACCEPT after all eight reported blockers were closed; the other eleven families remain contract-only or explicitly blocked until a dedicated typed runner is implemented and reviewed |

The next safe parallel wave assigns disjoint consumer/benchmark ownership as
follows. The coordination lane remains the only owner of shared catalogs,
planner/factory types, registry code, selectors, shared benchmark formats, and
this checklist.

| Proposed lane | Exclusive family-owned modules | Stop/return boundary |
| --- | --- | --- |
| Stimulus-epoch benchmark | family-local source/candidate reader matrix, focused fixtures/tests, and family doc | Use the integrated strict consumer; return instead of changing the writer, shared catalog, selector, or candidate profile |
| Chaser-distance benchmark | family-local sealed-base source/candidate reader matrix and focused tests | Do not change the sealed base, catalog, selectors, or shared publisher |
| Export dtype closure | cross-recording Arrow declarations and export-only tests | Do not change recording-local Zarr schemas or registry selection |
| Family benchmark adapters | one disjoint family adapter and evidence schema extension per lane | Return any shared runner/result-format change to coordination; never publish production artifacts |

Physical-profile promotion, production writer adoption, and selector changes
remain serialized policy work after representative evidence is reviewed.

Every implementation lane must satisfy this handoff checklist:

- [ ] Record the exact coordination base commit and branch/worktree path.
- [ ] Start from that commit in a new clean worktree; do not reuse an older dirty
      worktree merely because its name matches the family.
- [ ] Keep the worktree clean except for the declared ownership surface.
- [ ] Add exact positive, missing-field, unexpected-field, wrong-dtype,
      wrong-shape, and recomputed-digest tampering tests as applicable.
- [ ] Run focused tests outside the sandbox when real Zarr is involved; also
      run static compilation, Ruff, and `git diff --check`.
- [ ] Report the complete diff and validation evidence before committing.
- [ ] Commit one reviewable checkpoint without pushing, merging, rebasing, or
      changing the shared `/groups` checkout.
- [ ] Re-run the combined integration matrix after the reviewed commit is
      applied to the coordination branch.
- [ ] Leave production selectors and canonical archives unchanged until the
      corresponding correctness and performance promotion gates pass.

### Implemented checkpoints

The first two correctness phases and the initial catalog-closure checkpoint are
now implemented on the coordination branch:

- fail-closed completion, eligibility, swim-bout, eye-angle, visualization,
  and explicit legacy-selection boundaries: `bf98bea4`, `1f57eea6`, and
  `7c706daa`;
- manifest-exclusive immutable Parquet generations, short per-run advisory
  locking plus manifest compare-and-swap, exact part inventories, strict
  registry and consumer selection, and snapshot-bound downstream provenance:
  `f6f541b6`;
- run-parent initialization guard repair: `7d155423`.
- truthful catalog ownership and canonical parent mappings for tail-posture and
  bout-classification runs: `73b3888b`;
- strict completed/eligible swim-bout test fixtures for the catalog integration
  matrix: `209bc52e`;
- closed semantic classification initially covered 22 additional maintained,
  embedded, artifact, maintenance, active-legacy-shaped, and legacy surfaces:
  `a2668075` and `f178e315`; four top-level families have since moved into the
  central catalog, leaving 18 auxiliary surfaces;
- an explicit negative authority test proving the active legacy-shaped
  swim-bout-statistics writer cannot activate a selector: `f65713cf`;
- closed declaration grammar for stage IDs, import paths, constant names,
  artifact paths, policy identifiers, and direct entry points: `dd4c0f74`;
- a deterministic machine-readable coverage report joining the maintained
  stage registry, exact storage catalog, and classified auxiliary surfaces,
  including relational ownership and tamper validation: `fdc3f076`. Coverage
  schema v3 now records 14 maintained derived stages, 13 exact central
  contracts, 13 separate unpromoted candidates, and 18 auxiliary surfaces.
  Candidate run parents are explicit, so the dedicated chaser candidate parent
  cannot be confused with its production authority parent. Duplicate
  ownership, missing logical contracts, forged selector eligibility, forged
  promotion, and publication-mode drift fail closed.
- exact byte-planned tail-kinematics candidate publication with semantic fills,
  executable receipt replanning, direct/consolidated metadata equivalence,
  serial whole-shard ownership, and selector non-mutation: `600fab64`.
- selector-ineligible exact compact candidate publication for swim bouts and
  bout kinematics, including node-local rematerialization, per-array logical
  hashes, receipt validation, local direct/consolidated metadata equivalence,
  atomic run-group publication, authoritative-root reconsolidation after the
  final publisher metadata write, and unchanged authority pointers: begun in
  `d82dcf41` and hardened in the following coordination checkpoint.
- node-local sealed immutable publication for all ten maintained chaser
  component writers, with digest-bound writer receipts and no legacy pointer
  writes or same-name replacement: `fc6a48c5`.
- explicit self-digested chaser dependency handles and target-runner receipts
  that authorize only exact selector-ineligible components without `latest`
  discovery: `34c067e5` and `21e115b3`.
- selector-ineligible stimulus-response compact-v3 candidate creation through
  the shared byte planner/factory, with semantic fills, immutable publication,
  receipt replanning, direct/consolidated metadata validation, and unchanged
  parent pointers: `40434306`.
- deterministic read-only swim-bout/bout-kinematics source-versus-candidate
  benchmark matrices with rotated fresh processes, suite/receipt binding,
  direct/consolidated comparison, equality, object/byte/RSS/timing evidence,
  and explicit null physical-I/O telemetry when tracing is absent:
  `e348d53f`.
- one archive-wide publication/consolidation lock shared by every common atomic
  run publisher and every call through the shared consolidation helper, plus a
  post-tombstone visibility-repair hook. Controlled two-publisher interleaving
  and post-consolidation failure tests prove that the final consolidated view
  contains both successful runs or the exact failed/ineligible tombstone. A
  follow-up adversarial review found fork-state, direct-consolidator,
  family-specific activation-lock, and unverified repair gaps. The hardening
  checkpoint closes all inherited lock descriptors after fork, moves
  consolidation attrs inside the lock, migrates maintained
  stimulus/tail/bout/subject-mask metadata transactions, and binds repaired
  metadata to the exact pre-callback tombstone. Bout promotion and subject-mask
  authority activation now prove exact direct/inline metadata, including
  rollback and unknown-ack paths. Arbitrary legacy or external mutators remain
  safe only on a quiescent archive until migrated.

The integrated lifecycle/publication regression matrix passed 359 tests with 14
expected legacy compatibility xfails. A later independent adversarial review
identified the archive-lock gaps recorded above; the final focused adversarial
matrix passed 44 lock/atomic/bout/subject-mask tests, including the deterministic
open/register fork race, malicious tombstone rewrite, rollback consolidation, and post-commit
acknowledgement recovery. The exact-tabular/stimulus matrix passed 15 tests,
while a broader affected-stage run reached 52 passes without a failure before
unrelated long tail-compute cases were stopped. The exact
catalog/stage-array matrix passed 88
tests after integration. The combined exact-catalog, surface-classification,
and coverage-report matrix passed 103 tests. Physical-profile promotion and
production selector activation remain out of scope.

The tail lane passed 90 focused/broader storage regressions before integration;
the integrated tail/planner checkpoint passed 49 focused tests. Track schema
validation passed six exact-schema tests and the established 108-test motion
publication suite. The shared compact rematerializer and exact compact schemas
passed 24 focused tests. The compact candidate publisher passed seven focused
candidate tests, including authoritative consolidated opens, and nine existing
materializer tests. The combined rematerializer, publisher, and maintained
reader matrix passed 47 tests after the historical bout-kinematics fixture was
made explicit about completion, eligibility, and compatibility. These are
correctness results, not full-duration performance evidence.

The integrated chaser manifest/materializer/writer matrix passed 82 tests.
The remaining gaze/near-field and batch callers then passed 38 exact-handle and
fail-closed preflight tests in isolation. After integration with eye-angle and
the hardened atomic publisher, the combined focused matrix passed 86 tests.
Historical name/`latest` discovery remains available only through explicit
compatibility flags; it is not authorization to restore implicit discovery.

The integrated stimulus-response family matrix passed 142 tests. One unchanged
historical swim-bout fixture remains intentionally excluded because it creates
an unmarked v6 source that the current fail-closed swim-bout reader correctly
rejects; that same case fails on the stimulus lane's untouched base and is not
a v3 storage-candidate regression.

### Phase 0 — Preserve the audit baseline

- [x] Inventory the seven cataloged maintained derived families.
- [x] Identify derived stage-catalog entries outside the storage catalog.
- [x] Inventory companion analytics, chaser components, exports, and legacy
      writers.
- [x] Separate logical, physical, lifecycle, consumer, and benchmark concerns.
- [x] Add a machine-readable analytics coverage report generated from the live
      writer, stage, and storage catalogs.
- [x] Include selector-ineligible physical candidates and their exact run
      parents in coverage schema v3 without treating candidate presence as
      production adoption or promotion.
- [x] Make CI fail when a newly maintained array-bearing analytics stage has no
      explicit coverage classification.

### Phase 1 — Fix fail-closed lifecycle behavior

- [x] Change generic workflow availability to use the canonical completion and
      eligibility resolver.
- [x] Reject selector-ineligible candidates during implicit reuse.
- [x] Permit statusless or lexical legacy discovery only behind an explicit
      inspection/compatibility policy.
- [x] Add crash-window tests for completion written before activation.
- [x] Migrate swim-bout run discovery to exact completion and eligibility
      checks.
- [x] Migrate eye-angle readers and overlay tooling away from unqualified
      `latest` lookup.
- [x] Confirm explicit run selection fails terminally instead of falling back.

### Phase 2 — Make export generations atomic

- [x] Write each Parquet export generation into a hidden temporary sibling.
- [x] Validate the complete part inventory, schemas, row counts, and content
      digests before publication.
- [x] Atomically rename the validated generation into its immutable final path.
- [x] Make the manifest enumerate the only allowed part files.
- [x] Reject extra, missing, duplicate, or digest-mismatched parts.
- [x] Change consumers to read the manifest-selected files rather than globbing
      a directory.
- [x] Require validation before registry indexing or activation.
- [x] Add a regression test in which a replacement generation has fewer parts.

### Phase 3 — Complete the executable analytics catalog

- [x] Add `tail_posture_view` with its exact parent, schema, method, publication
      owner, physical owner, registry mode, and planner status.
- [x] Add `bout_classification` with the same executable declarations.
- [x] Decide and record whether each of the following is a maintained
      scientific authority, an embedded component, a visualization/cache, an
      export, maintenance output, or legacy:
    - [x] stimulus epochs;
    - [x] detection/session occupancy;
    - [x] chaser-distance base;
    - [x] each chaser component in the versioned analysis profile;
    - [x] registered-detection gate/QC outputs;
    - [x] speed runs;
    - [x] swim-bout statistics;
    - [x] in-place chaser-state interpolation.
- [x] Add canonical run-parent mappings for currently cataloged maintained
      families.
- [x] Move stimulus epochs, detection/session occupancy, and the sealed
      chaser-distance base from temporary auxiliary classification into exact
      central stage/storage/candidate/registry ownership. Keep the ten chaser
      components as embedded independently sealed surfaces.
- [x] Remove tail-posture and bout-classification availability-only local parent
      declarations after catalog adoption.
- [x] Add tests proving the currently cataloged stage catalog, storage catalog,
      run-parent catalog, and materializer ownership agree.

### Phase 4 — Freeze exact logical schemas

- [x] Introduce or reuse one exact analytics stage-schema representation that
      binds concrete paths to versioned `ArrayContract` identities.
- [ ] Require exact dtype, axes, shape, units, coordinates, null/fill semantics,
      access class, mutability, and authority role for every array.
- [x] Freeze a new opt-in compact stimulus-response v3 schema with 19 exact
      tables, up to 310 unique arrays, fixed dtypes/string widths, and closed
      required/all-or-none optional bundles.
- [x] Reject data-dependent schema expansion, silent multidimensional field
      loss, and lossy text truncation in compact v3 publication. Production v2
      remains unchanged behind its compatibility/default boundary until a v3
      canary and promotion gate pass.
- [x] Complete exact dtype validation for every eye-angle semantic array.
- [x] Add closed whole-run manifests for swim bouts and bout kinematics.
      Swim-bout v8 binds 132 required arrays plus its optional embedded frame
      axis; bout-kinematics v7 binds 111 required arrays plus an optional
      45-array eye-gaze bundle. Both validate before publication and require an
      explicit compatibility mode for historical layouts.
- [x] Freeze exact per-field bout-classification storage declarations.
      Maintained schema v2 contains 20 exact arrays, fixed 64/128-byte text
      matrices, an executable manifest, and explicit v1 compatibility.
- [x] Bind all ten tail-posture arrays to explicit analytics contracts.
      Schema v3 records the still-current subject-mask row-chunk helper as a
      compatibility physical owner with `byte_planner_adopted=false`; migrating
      that physical owner remains Phase 6 work.
- [x] Freeze tail kinematics as an exact 21-array core plus an all-or-none
      two-array source-revision bundle, including fixed dtypes, symbolic axes,
      roles, access classes, write modes, and semantic fills.
- [x] Freeze track kinematics as an exact 69-array per-track core, closed
      35-array physical bundle, required/optional run identities, and explicit
      legacy exclusions without weakening float64 coordinate authority.
- [x] Resolve the derived-keypoint-metric ownership contradiction: the embedded
      `derived_metrics_schema` triangle surface is legacy v1 compatibility;
      maintained v2 diagnostics belong to versioned `keypoint_quality_runs`
      profiles and do not weaken the exact refined-keypoint-v2 inventory.
- [x] Freeze detection occupancy as a 30-array epoch authority and session
      occupancy as a distinct 29-array full-recording authority. The exact
      inventories differ only by the required stimulus-window lineage column;
      cross-family inventories fail closed.
- [x] Freeze the chaser-distance candidate as the exact 30-array union protected
      by its publication seal, epoch-window authority, and surface manifest.
      Unsealed protocol-role, raw-count, threshold-fraction, visualization, and
      derived-component arrays are explicitly outside this base projection.
- [x] Record the fail-closed track adoption blocker: two structured lineage
      dtypes cannot yet round-trip through `DTypeContract`, `StoragePlan`, the
      array factory, and physical metadata comparison.
- [x] Resolve that blocker with an explicit versioned representation change:
      the candidate projects the two v1 structured lineage records into five
      primitive v2 arrays and reconstructs the v1 records only through an
      explicit compatibility reader. Candidate/source hashes, exact source
      paths, and complete group/array topology are fail-closed.
- [ ] Define exact Arrow dtypes for export columns where cross-language stability
      requires them.
    - [x] Freeze and enforce the exact ordered 62-field
          `position_occupancy_histogram_2d` schema through writer, staged
          publication, and manifest-selected validation.
    - [x] Freeze the exact ordered 32-field `recording_summary` schema, including
          stable null columns for absent source capabilities and manifest-exact
          zero-row generations with no placeholder Parquet part.
    - [x] Freeze the exact ordered 95-field `baseline_behavior_summary` schema,
          including the closed eight-key source-summary projection, nullable
          FPS discrepancy, and representation-only authority boundary.
    - [x] Freeze the exact ordered 77-field `baseline_behavior_time_bins` schema,
          including its closed 38-key metric vocabulary and exact selected
          baseline-strategy consumer boundary.
    - [x] Freeze the exact ordered 71-field `baseline_kinematic_samples` schema,
          including its closed 32-key sample vocabulary, null-not-sentinel
          semantics, and exact selected baseline-strategy consumer boundary.
    - [x] Freeze the exact ordered 60-field `stimulus_steps` schema, including
          closed maintained moving/concentric child vocabularies and rejection
          of dynamic compatibility columns.
    - [x] Freeze the exact ordered 38-field `stimulus_step_summary` schema and
          correct its grain/key to recording × fish × stimulus step.
    - [x] Freeze the exact ordered 129-field
          `stimulus_response_per_fish_step` nullable mode union. Arbitrary
          source attributes no longer expand the schema; only the two named
          OMR method-version attributes are projected.
    - [x] Freeze the exact ordered 70-field `swim_bout_metrics` schema with
          candidate/signal identity, exact primary-key enforcement, and
          explicit legacy source compatibility.
    - [x] Freeze the exact ordered 150-field `bout_kinematics_metrics` schema.
          Its grain and primary key now include measurement level, storage-only
          fixed-text suffixes are removed from the query representation, and
          unknown measurement families fail closed.
    - [ ] Replace inferred schemas only after producer semantics and nullability
          are frozen. The canonical envelope currently has 12 exact and 18
          inferred tables; the remaining inferred tables are the chaser
          exports. Both group-statistics tables use exact schemas. Baseline strategy
          consumes exact canonical baseline inputs but still writes four
          separately inferred output tables; training response likewise has
          three inferred output tables.
- [x] Add recomputed-digest tampering and unexpected-field tests for the eye,
      bout-classification, and tail-posture manifests completed so far.
- [ ] Add equivalent adversarial coverage for every remaining new manifest.
      Stimulus-response, swim-bout, and bout-kinematics now have this coverage;
      later families remain.

### Phase 5 — Seal chaser components independently

- [x] Define an immutable component manifest envelope with source-run identity,
      schema, parameters, array declarations, content digests, and completion.
- [x] Give each component its own hidden-copy/validate/rename publication step.
      All ten maintained writers now use the shared sealed staging and atomic
      component publisher.
- [x] Remove delete-and-rewrite publication of visible components from the ten
      maintained writer paths.
- [x] Define and test component selectors bound to exact manifest digests. Keep
      production activation quarantined until atomic workflow adoption.
- [x] Record every requested component's output identity, exact dependency
      handle, validation result, and selector-ineligible explicit-authority
      state in the chaser runner receipt. Individual writer receipt v2 returns
      the handle, and cluster target receipt v2 embeds a self-digested receipt
      rebuilt from the read-only authoritative archive after all steps finish.
- [x] Migrate the egocentric-bearing -> bout-response -> escape-events chain to
      explicit selector-ineligible dependency handles and bind upstream
      manifest digests into downstream lineage. Historical swim-bout input is
      permitted only through an explicit compatibility flag.
- [ ] Migrate the remaining chained scientific consumers and exports to exact
      component handles.
- [ ] Keep export of unsealed component tables fail closed.
- [ ] Add recovery tests for interrupted component publication and stale
      pointers. Post-selector failure rollback is covered; add interruption
      points for each adopted component workflow.

The v1 logical envelope, validation boundary, and remaining adoption checklist
are frozen in `docs/chaser_component_publication_contract_v1.md`.

### Phase 6 — Adopt byte-aware physical planning

- [x] Implement the strict declaration/facts adapter from exact
      `AnalysisArrayDeclaration` records into the shared
      `ArrayIntent -> StoragePlan` planner. It preserves complete trailing
      record axes, derives rows from fixed-width dtype bytes, restricts shards
      to the declared growth axis, and emits a digest-bound plan and object
      estimate. This is infrastructure only; no production profile or writer
      is promoted by its existence.
- [ ] Define an `ArrayIntent` for every maintained analytics array.
- [ ] Classify access as eager, windowed, per-row, indexed, bulk-scan, or
      artifact-byte-stream.
- [ ] Classify write mode as immutable, append/whole-shard-owned, or editable.
- [ ] Derive chunks from dtype and per-row/per-unit shape, not a universal row
      count.
- [ ] Derive shards from target bytes and immutable whole-shard ownership.
- [ ] Pin Zarr v3 and the exact bytes/compression/checksum/index codec chain in
      the storage profile and manifest.
- [ ] Preserve independently readable inner chunks inside shards.
- [ ] Record requested/effective chunks and shards plus object estimates.
- [x] Add planner parity tests for short recordings, million-frame recordings,
      empty tables, wide matrices, and fixed-width text arrays.

Candidate-adoption checkpoints (these do not promote a profile):

- [x] Eye-angle compact-v7 candidate writes all 41 arrays through a recomputed
      plan receipt and remains selector-ineligible.
- [x] Eye-angle candidate metadata uses a closed path-aware fill contract:
      NaN for invalid/unavailable float payloads, false for boolean
      availability/validity, and zero for QA, text, and mandatory coordinates.
- [x] Tail-posture v3 candidate writes all ten arrays through the shared
      planner/factory while preserving the default writer and selectors.
- [x] Bout-classification v2 candidate writes all 20 arrays through the shared
      planner/factory while preserving the default writer and selectors.
- [x] Tail-posture and bout-classification candidate metadata freeze semantic
      NaN, `-1`, false, and zero fills rather than using one numeric default.
- [x] Tail-kinematics candidate writes the exact 21-array core and optional
      revision bundle through the shared planner/factory, persists a
      digest-bound receipt, validates direct/consolidated declarations, rejects
      process-shard ownership, and leaves all parent selectors unchanged.
- [x] Subject-shape candidate adoption. The access-aware candidate validates
      the producer-sealed source before planning, preserves the complete v4
      inventory and path-specific fills, writes through the shared factory,
      publishes atomically without pointer changes, and requires persisted
      direct/consolidated subtree equivalence.
- [x] Track-kinematics candidate adoption. The selector-ineligible v2 layout
      preserves float64 positions, writes all primitive arrays through the
      shared planner/factory, binds source and candidate logical hashes, proves
      full direct/consolidated subtree equivalence, and publishes atomically.
- [x] Shared columnar adoption for swim bouts and bout kinematics. The new
      candidate publisher rematerializes exact compact runs through the shared
      byte planner without changing either production writer or selector.
- [x] Add the shared exact compact rematerialization boundary used by that
      adoption: it derives the growth axis from semantic axes, creates arrays
      through the common factory, writes complete physical units, preserves
      explicit report artifacts, and validates an executable receipt.
- [x] Compact stimulus-response v3 candidate adoption. The opt-in writer and
      materializer use the shared planner/factory, pin the existing published
      HTTP candidate profile, enforce serial whole-shard ownership, validate a
      digest-bound plan and current metadata-equivalence receipt, publish
      immutably, and leave selectors unchanged.
- [x] Detection/session occupancy opt-in candidate adoption. The existing
      direct writer now stamps a digest-bound exact manifest, while the shared
      compact rematerializer derives chunks and shards from bytes, validates
      decoded equality and the complete direct/consolidated subtree, publishes
      atomically, and leaves the source selector unchanged.
- [x] Chaser-distance sealed-base candidate adoption. The candidate derives
      its exact 30-array inventory independently from the canonical v1 source
      authorities, binds all source record digests and decoded hashes, writes
      the byte-planned projection atomically, validates persisted
      direct/consolidated metadata, and leaves both source and candidate-parent
      selectors unchanged.

Candidate creation is complete for the thirteen centrally cataloged families.
Recommended benchmark and promotion-review order:

1. eye angles;
2. tail kinematics;
3. subject shape;
4. tail-posture view and bout classification;
5. track kinematics;
6. shared columnar storage, thereby covering swim bouts and bout kinematics;
7. stimulus-response producer/consumer benchmarks;
8. stimulus-epoch, occupancy, and chaser-distance families.

### Phase 7 — Complete publication and registry projection

- [x] Add serialized registry completion/invalidation projection for all thirteen
      maintained catalog stages. The finalizer reopens each archive directly,
      requires matching `latest`/`latest_complete`, completion, and selector
      eligibility, and dispatches registry writes serially; no worker owns the
      SQLite transaction.
- [ ] Decide production selection policy for body frame and keypoint quality;
      do not infer activation from their selector-ineligible canaries.
- [ ] Standardize copy-integrity policy across maintained families.
- [x] Require direct and consolidated metadata equivalence before serialized
      registry visibility. The generic subtree comparator reads persisted JSON
      rather than accepting Zarr's direct-metadata fallback.
- [x] Ensure the serialized finalizer binds completion, eligibility, selector,
      receipt, declaration digest, and registry state after recovery. Requests
      are ordered by the stage DAG so one batch cannot invalidate its own
      downstream completions.
- [x] Add idempotent finalizer and partial-commit retry tests. Exact replays are
      reported as `already_published`; different evidence cannot rebind the
      same selected run.

### Phase 8 — Implement declared workflow outputs

- [x] Add the stimulus-response execution adapter. It explicitly requests the
      selector-ineligible compact-v3 materializer path and binds the selected
      stimulus, track-kinematics, and swim-bout dependencies.
- [x] Add dependency-bound execution adapters for tail-posture view and bout
      classification. Workflow-profile adoption remains a separate policy
      choice; registering a command builder does not activate either stage.
- [ ] Implement exact contracts and publishers for:
    - [ ] sampled kinematic exports;
    - [ ] activity/spatial summaries;
    - [ ] eye traces;
    - [ ] tail traces.
- [ ] Keep recording-local Zarr authorities separate from immutable
      cross-recording Parquet query products.
- [ ] Bind each export to exact selected recording-local manifests.

### Phase 9 — Benchmark and promote one family at a time

- [x] Add a closed benchmark-coverage catalog with exactly one record per
      physical candidate. It distinguishes plan-only families from executable
      read matrices and records writer, publication, physical-I/O, Palette,
      required Crimson, short-scale, and full-scale evidence independently.
      Measured/executed claims require an immutable receipt digest and
      versioned passing gate. All thirteen current families now resolve an
      executable read matrix; none yet claims complete benchmark coverage, and
      the catalog never serves as profile-promotion authorization.
- [x] Add a deterministic suite planner that binds every array workload and
      the whole-run publication workload to the exact logical declaration,
      storage-plan receipt, logical dimensions, seed, and safety policy. It
      covers 200,000- and 1,000,000-row contract scales and rejects rehashed
      plan, selection, publication, or eligibility-policy tampering.
- [ ] Define a deterministic writer/publisher/reader workload for every
      maintained family.
- [x] Implement the deterministic read-side source/candidate matrix for exact
      swim-bout and bout-kinematics candidates. It exercises receipt-bound
      access selections and full scans in rotated fresh processes. Publication
      timing remains unmeasured: opaque publisher provenance is not treated as
      replayable timing evidence. Physical request and transfer counts remain
      explicitly unavailable until tracing is added.
- [x] Implement the family-local sealed chaser-distance source/candidate matrix.
      It binds the exact source authority, candidate manifest, storage receipt,
      atomic publication owner, full decoded values, access-class selections,
      and direct/consolidated metadata across rotated fresh processes. It is
      explicitly nonpromoting and leaves physical request/transfer telemetry
      unavailable until genuine tracing is added.
- [x] Implement the family-local stimulus-epoch v1-source/v2-candidate matrix.
      It embeds and deeply cross-binds the canonical source and candidate
      lineage payloads, executable storage receipt, and complete run manifest;
      coordinated rehashes fail offline validation. Rotated child processes
      prove complete decoded-array and segment equality without changing the
      current v1 authority or promoting the v2 candidate.
- [x] Implement the family-local stimulus-response compact-v3 source/candidate
      matrix. It validates the strict maintained reader and every declared
      array, reconstructs the HTTP-v1 plan offline, rotates fresh child
      processes, records null physical-I/O fields until external tracing exists,
      and requires an externally pinned outer evidence digest before later gate
      claims.
- [x] Implement the family-local subject-shape v4 source/candidate matrix. It
      embeds normalized source/candidate declarations, reconstructs the
      candidate physical plan and installed transform exemptions offline,
      binds every child to an exact driver/role/order position, rejects
      component symlinks before reads, and remains explicitly nonpromoting.
- [x] Implement the family-local eye-angle compact-v7 source/candidate matrix.
      It validates all 41 arrays, executable schemas and physical declarations,
      exact nested publication/materialization authority and logical-input
      roles, rotated fresh processes, archive guards, and hard nonpromotion.
- [x] Implement the family-local track-kinematics v1-source/v2-candidate
      no-physical matrix. It binds the public v1 reader and diagnostic-only v2
      adapter to exact declarations, storage and publisher receipts, live
      selectors, matched source/candidate logical workloads, fresh processes,
      and replay against the live immutable archive. The optional 35-array
      physical bundle is explicitly not covered.
- [x] Implement the family-local tail-kinematics v2 source/candidate matrix.
      It closes the exact 21-array core and atomic 23-array revision bundle,
      pins the complete registered `published_http_v1` profile, independently
      reconstructs stable scientific identity from both runs, binds the real
      atomic publisher/staged-source authority, and rejects coordinated
      re-signing while keeping publisher copy facts explicitly non-replayable.
- [x] Implement the family-local tail-posture-view v3 source/candidate matrix.
      It enforces exact ten-array row semantics (including canonical reasons,
      finite valid rows, all-NaN invalid rows, and radian/degree agreement),
      exact profile and scientific identity, balanced fresh processes, live
      selector/metadata replay, honest diagnostic-only payload adapters, and
      self-validation before immutable matrix evidence is written. The full
      Megabouts input-pack consumer remains a separate dependency-complete
      gate.
- [x] Implement the family-local bout-classification v2 source/candidate
      matrix. It freezes all twenty arrays and semantic sentinels, binds the
      complete tail/shape/track/swim dependency identity, enforces classified
      HB1 frame arithmetic, pins the exact registered profile, and exercises
      first/middle/final row windows plus eager reads in balanced fresh
      processes. The candidate reader remains truthfully diagnostic-only.
- [x] Register all thirteen read matrices in the executable coverage catalog.
      Catalog tests resolve every adapter while leaving writer, publication,
      physical-I/O, Palette/Crimson consumer, representative-scale, and
      promotion fields false and unbound.
- [x] Freeze the shared writer/publication execution request and successful
      receipt contract without authorizing production selection. All thirteen
      adapters bind a family-specific equality projection and coordinate role;
      requests require the live registered adapter, exact candidate profile,
      exact live family-suite projection, a recognized node-local scratch root,
      immutable production pre-state probes, and a benchmark-only nonmutation
      policy.
      Receipt v2 represents completed publications only, requires all eleven
      measured phases, externally anchors the request digest, distinguishes
      application counters from genuine filesystem/network transfer, and
      rejects zero-byte or coordinate-minting claims. Failed attempts require a
      separate runner-owned attempt record rather than a dishonest partial
      success receipt.
- [x] Bind every runner-affecting argument in a closed invocation envelope as
      `fbbce911`. Request, receipt, and failed-attempt envelopes are now honest
      v2 contracts: each rejects legacy v1 and numeric schema-version aliases.
      The first frozen parameter grammars cover exact-tabular, eye-angle, and
      track-flat execution. Implemented adapters cannot select a contract that
      still lacks an exact parameter grammar, and the exact-tabular child/CLI
      no longer accepts unsigned copy-backend or scratch-retention flags.
      Track-flat publication is explicitly scoped to the existing
      `analysis/track_kinematics_runs/offline` namespace; the broad family
      parent remains `analysis/track_kinematics_runs`, whose independent online
      and offline scopes are semantic sources rather than matched layout pairs.
      Independent adversarial review is ACCEPT, and the complete focused gate
      passed 73 tests outside the sandbox.
- [x] Implement the first shared typed runner for swim bouts and bout
      kinematics as `71d13675`. It requires immediate immutable run children,
      resolves the live public track-motion authority, resolves the swim-bout
      frame axis, proves bout kinematics uses the same authority as its exact
      swim-bout source, stages on node-local scratch, rematerializes through the
      byte planner, and keeps all acceptance inside the atomic publication
      boundary. The driver independently binds child PID/start identity and
      collision-resistant registry/profile tree snapshots before exposing a
      success receipt; malformed child evidence, protected-state changes, and
      receipt-publication failures produce immutable attempt evidence and
      tombstone the exact owned candidate. Independent re-review is ACCEPT.
      The complete 67-test post-fix execution, runner, materializer, telemetry,
      and atomic-publication suite passed.
- [ ] Implement and independently review dedicated typed runners for the other
      eleven catalog families. Do not infer executability from a contract-only
      catalog entry.
    - [x] Eye angles (`159742e6`) now execute the frozen request-v2 invocation,
          reconstruct and validate the exact 41-array compact-v7 suite, bind
          live subject-shape and canonical-keypoint authorities, measure all
          eleven phases, and keep acceptance, selector ineligibility, protected
          post-state observation, and terminal tombstoning inside the typed
          execution boundary.
    - [x] Track kinematics (`15d5e0a7`) now executes the primitive flat-v2
          migration candidate from the current structured v1 source. The
          source/candidate dual validation is migration evidence, not a policy
          to retain structured dtypes in new writers. The diagnostic suite
          excludes the optional 35-array physical-coordinate bundle and
          therefore remains deliberately nonminting and unable to pass the
          publication gate.
    - [x] Activate both typed runners in the shared executable catalog and
          remove test-local catalog substitutions. Track success and failure
          now traverse the real parent driver and fresh child, publish exactly
          one receipt or attempt sidecar, preserve exact child error identity,
          and re-observe selector, registry, and production-profile state
          before exposing terminal evidence. The live eye runner performs an
          exact 41-array compute/rematerialize/publication pass; its untraced,
          dirty-worktree unit receipt correctly remains nonpromoting. The
          combined shared/eye/track execution, materializer, recovery, and read
          matrix gate passed 136 tests outside the sandbox.
    - [ ] Implement the remaining nine typed family runners.
- [ ] Measure node-local compute, validation, consolidation, copy, and atomic
      publication separately.
- [ ] Measure apparent/allocated bytes, object count, compressed transfer,
      latency distributions, throughput, CPU, and peak RSS.
- [ ] Exercise real access patterns: eager small arrays, random frame/row,
      windowed playback, indexed ranges, full scans, and exports.
- [ ] Include representative short (~200,000-frame) and full-duration
      (~1,000,000-frame) recordings where the row axis is frame-like.
- [ ] Validate exact decoded equality between old and candidate layouts.
- [ ] Run Palette readers and Crimson adapters where the family is user-facing
      in Crimson.
- [ ] Promote a versioned physical profile only when correctness gates pass and
      the candidate materially improves its intended workload without a
      significant regression elsewhere.
- [ ] Retain the previous profile as an explicit rollback reader until the
      migration window closes.

### Phase 10 — Retire compatibility debt deliberately

- [ ] Mark each legacy writer and reader with an explicit compatibility status.
- [ ] Prevent legacy outputs from becoming implicit current authority.
- [ ] Provide migration/republication tools where historical scientific value
      justifies them.
- [ ] Remove legacy fallbacks only after inventorying affected archives and
      proving supported replacements.
- [ ] Update the canonical analytics matrix from executable catalogs rather than
      copying version numbers by hand.

## Promotion Gates

No analytics physical profile or selector should be promoted until all relevant
gates pass:

- exact logical schema and dtype validation;
- source-lineage and identity validation;
- direct/consolidated metadata equivalence;
- immutable atomic publication and recovery tests;
- selector/registry agreement;
- exact decoded equality against accepted source data;
- bounded object count and request amplification;
- real-consumer correctness and cancellation behavior;
- benchmark evidence for both representative short and long recordings; and
- an explicit versioned rollback path.

## Immediate Next Checkpoint

Continue consumer closure and measurement without promoting defaults:

1. extend deterministic per-family writer/rematerialization/publication/read
   benchmarks, first for stimulus response/epochs, occupancy, subject shape,
   track kinematics, and chaser distance;
2. run representative short and full-duration matrices with physical I/O,
   object count, phase timing, CPU, and RSS evidence;
3. close remaining chaser component consumer/recovery and cross-recording Arrow
   dtype gaps in isolated worktrees, returning shared-format changes to this
   coordination lane; and
4. review promotion one family at a time, retaining the current production
   writer/profile and selector as the rollback boundary until every gate passes.
