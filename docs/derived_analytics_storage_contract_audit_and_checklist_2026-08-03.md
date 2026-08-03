# Derived Analytics Storage Contract Audit And Implementation Checklist

Date: 2026-08-03

Status: active implementation checklist; reconciled through coordination
checkpoint `d82dcf41`; correctness, executable-catalog, shared-planner,
serialized-registry, compact-rematerialization, and benchmark-planning
foundations are integrated; selector-ineligible swim-bout and bout-kinematics
candidate publishers are integrated; opt-in writer candidates are being
adopted family by family; no production profile or scientific selector
authority changed by these checkpoints

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

Nine maintained array-bearing derived families now have executable catalog
entries. Seven use the shared atomic materializer boundary:

- track kinematics;
- swim bouts;
- bout kinematics;
- eye angles;
- subject shape;
- tail kinematics; and
- stimulus response.

Tail-posture views and bout classification are also cataloged, truthfully, as
guarded direct writers rather than falsely claiming atomic-materializer
ownership. None of the nine uses the shared byte-budgeted planner as its
production default yet. Tail-posture, bout-classification, tail-kinematics,
eye-angle, swim-bout, and bout-kinematics now expose explicit
selector-ineligible planner candidates. All nine have serialized registry
projection for eligible authorities. Stimulus epochs, occupancy outputs, and
the chaser analysis suite use additional stage-local schemas and writers.
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
| Track kinematics | Exact 69-array per-track core, closed 35-array physical bundle, run identities, legacy exclusions, materializer, and registry contract | Version or flatten the two structured lineage dtypes in the shared factory before candidate adoption; retain float64 position authority |
| Subject shape | Strong typed semantic surface, content manifest, lineage, atomic sharded publication, and serialized registry projection | Close the currently dynamic component/relation/body-frame array inventory before byte-planned adoption |
| Tail kinematics | Exact 21-array core plus an all-or-none two-array revision bundle, coordinate/lineage semantics, atomic publication, registry projection, and an explicit selector-ineligible byte-planned candidate | Run full-duration producer/reader benchmarks before profile promotion |
| Eye angles | Exact 41-array compact-v7 schema, deep manifests, strict maintained readers, registry projection, and an explicit selector-ineligible byte-planned direct-writer candidate | Add a consolidated atomic publication boundary, then benchmark through Palette and Crimson before profile promotion |
| Swim bouts | Exact compact-v8 whole-run array manifest, authoritative selection, serialized registry projection, and an immutable selector-ineligible byte-planned candidate publisher | Benchmark publication and the real consumer before profile promotion |
| Bout kinematics | Exact compact-v7 manifest, authoritative selection, serialized registry projection, and an immutable selector-ineligible byte-planned candidate publisher | Benchmark publication and the real consumer before profile promotion |
| Stimulus response | Exact opt-in compact-v3 schema with closed table bundles, fixed dtypes, strict coercion, and atomic publication; legacy v2 remains the default compatibility path | Adopt byte planning in the v3 writer and benchmark before any default change |
| Body frame | Exact ten-array schema, byte planner, strict manifest, and consolidated publication | Move from selector-ineligible companion evidence to an explicitly governed production lifecycle when authorized |
| Keypoint quality | Exact diagnostic schema, byte planner, strict manifest, and consolidated publication | Add explicit production lifecycle/registry policy when authorized |
| Derived keypoint metrics | Current v2 keeps profile-specific diagnostics in immutable source-bound `keypoint_quality_runs`; refined v2 retains compact acceptance gates and forbids legacy triangle arrays | Add any future skeleton-specific triangle metrics as a versioned quality profile, not universal refined-keypoint arrays |
| Tail-posture view | Exact typed arrays, guarded lifecycle, and an explicit selector-ineligible byte-planned candidate with semantic fills | Benchmark the candidate and decide whether to migrate to the atomic materializer |
| Bout classification | Exact compact-v2 manifest, guarded activation, serialized registry projection, and an explicit selector-ineligible byte-planned candidate with semantic fills | Benchmark the candidate through its real consumer |
| Track visualization | Shared PNG/spec byte artifact contract | Keep classified as an artifact rather than a numeric scientific run |
| Stimulus epochs | Explicit typed window columns | Add catalog ownership, byte planning, manifest validation, and atomic publication |
| Detection/session occupancy | Scientific schema IDs and direct writers | Freeze exact arrays and lifecycle; adopt planner and atomic publisher |
| Chaser-distance base | Hardened immutable base and guarded activation | Add to the central analytics catalog and byte planner |
| Chaser components | Protocol-neutral schemas, payload-bound manifests, verified detached reads, and all ten maintained writers routed through node-local sealed immutable publication | Add explicit digest-bound dependency handles, runner receipts, consolidated activation, export adoption, and chained recovery coverage |
| Cross-recording Parquet exports | Versioned table/manifest contracts with immutable, manifest-exclusive atomic generations | Freeze exact Arrow dtypes where cross-language stability requires them |
| Core workflow exports | Declared planning nodes for sampled kinematics, summaries, eye traces, and tail traces | Implement adapters and exact export contracts |
| Legacy/in-place outputs | `speed_runs`, swim-bout statistics, and stimulus/chaser mutation paths remain | Classify as legacy/maintenance or migrate; do not leave them implicitly current |

## Evidence And Known Disagreements

The executable catalog explicitly says physical policy remains stage-owned and
exposes `byte_planner_adopted` as the migration boundary
(`src/fisheye/analysis_workflows/storage_contract_catalog.py:1-10,20-35`). Its
nine entries are at
`src/fisheye/analysis_workflows/storage_contract_catalog.py:208-348`.

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
nine-family catalog and remains incomplete as a full writer census. The earlier
reconciliation
correctly leaves the per-array inventory, byte-planner migration, and consumer
benchmarks open
(`docs/derived_analytics_storage_reconciliation_2026-08-01.md:102-124`).

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
selector-ineligible immutable candidates. The remaining correctness boundary
is consumption and orchestration: historical chained workflows try to
rediscover a just-written candidate through legacy `latest`, while exports and
runner receipts do not yet carry an explicit validated component handle.
Candidates remain intentionally undiscoverable until those consumers bind an
exact component manifest digest or a separately reviewed selector activation.

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
start from the current clean coordination history (`d82dcf41` at this
reverification), not from one of the older worktree heads listed by
`git worktree list`. A historical dirty worktree is evidence to reconcile, not
a safe base for new work.

| Lane | Worktree / branch | Base commit | Owned implementation surface | Integration rule |
| --- | --- | --- | --- | --- |
| Coordination | repository root / `agent/palette/derived-analytics-storage-contracts-20260803` | through `d82dcf41` | Shared catalogs, checklist, cross-family tests, reviewed cherry-picks | Serial owner; never rebased onto an unreviewed lane |
| Surface classification | `/tmp/palette-analytics-surface-classification-20260803` / `agent/palette/analytics-surface-classification-20260803` | `4923757f` | Closed classification catalog and its focused tests | Integrated as `a2668075` and `f178e315`; it classifies rather than promotes |
| Eye-angle exact schema | `/tmp/palette-eye-v7-clean-20260803` / `agent/palette/eye-v7-clean-20260803` | `dd4c0f74` | Reusable analytics array declaration, exact 41-array eye-angle schema, deep writer/reader/readiness validation, and explicit v2-v6 compatibility boundaries | Reviewed as ready, committed as `3025d66e`, and integrated as `c1976300` |
| Eye-angle byte candidate | historical isolated lane | `28f0132c` through `ac6b8bc3` | Exact 41-array planner/factory candidate and semantic fills | Integrated; remains explicit and selector-ineligible |
| Tail-kinematics byte candidate | historical isolated lane | integrated as `600fab64` | Exact 21-array core, revision bundle, planner/factory candidate, metadata equivalence, and lifecycle guards | Integrated; remains explicit and selector-ineligible |
| Track-kinematics exact schema | historical isolated lane | integrated as `af38edba` | Exact current motion vocabulary and explicit shared-factory blocker | Integrated; no physical candidate was created |
| Shared compact rematerialization | coordination lane | integrated as `5d7fb30c` | Exact declaration-bound replanning, frame-axis growth, whole-physical-unit writes, receipt and metadata validation | Infrastructure only; family adoption remains explicit and opt-in |
| Shared compact candidate publication | coordination lane | integrated from `d82dcf41` | Immutable selector-ineligible swim-bout and bout-kinematics candidates, logical hashes, local and authoritative-root metadata equivalence, and selector non-mutation | Candidate evidence only; benchmark and promotion remain pending |
| Chaser scientific writer adoption | historical isolated lane | integrated as `fc6a48c5` | All ten component writers, sealed staging capability, atomic component publisher, focused tests, and lifecycle doc | Integrated as candidate publication; explicit dependency handles and activation remain separate |

The next safe parallel wave assigns disjoint ownership as follows. The
coordination lane remains the only owner of shared catalogs, planner/factory
types, registry code, selectors, and this checklist.

| Proposed lane | Exclusive family-owned modules | Stop/return boundary |
| --- | --- | --- |
| Stimulus-response v3 byte candidate | stimulus-response writer, exact v3 schema, materializer, reader, focused tests, and family doc | Return instead of changing shared planner/factory or catalog declarations |
| Subject-shape exact closure | subject-shape schema/writer/materializer, focused tests, and family doc | Freeze supported component bundles before creating candidate arrays; the current dynamic inventory is an explicit blocker |
| Chaser component adoption | chaser component writers, runner receipt, focused interruption tests, and family doc | Use the existing atomic component publisher; return instead of changing selectors/catalogs |

After those lanes reconcile serially, one shared-columnar lane may own both
swim-bout and bout-kinematics adoption. Those two families must not be split
across concurrent lanes because both depend on the same columnar creation
boundary.

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
- closed semantic classification of 22 additional maintained, embedded,
  artifact, maintenance, active-legacy-shaped, and legacy surfaces:
  `a2668075` and `f178e315`;
- an explicit negative authority test proving the active legacy-shaped
  swim-bout-statistics writer cannot activate a selector: `f65713cf`;
- closed declaration grammar for stage IDs, import paths, constant names,
  artifact paths, policy identifiers, and direct entry points: `dd4c0f74`;
- a deterministic machine-readable coverage report joining the maintained
  stage registry, exact storage catalog, and all 22 classified auxiliary
  surfaces, including relational ownership and tamper validation:
  `fdc3f076`.
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

The integrated lifecycle/publication regression matrix passed 359 tests with 14
expected legacy compatibility xfails. The atomic-publication lane also received
an independent adversarial review with no remaining correctness or
publication-safety blockers. The exact catalog/stage-array matrix passed 88
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
Twenty broader historical chained-workflow cases remain intentionally failing
closed because they try to select an ineligible component through `latest`;
they are the explicit-handle migration surface, not authorization to restore
implicit discovery.

### Phase 0 — Preserve the audit baseline

- [x] Inventory the seven cataloged maintained derived families.
- [x] Identify derived stage-catalog entries outside the storage catalog.
- [x] Inventory companion analytics, chaser components, exports, and legacy
      writers.
- [x] Separate logical, physical, lifecycle, consumer, and benchmark concerns.
- [x] Add a machine-readable analytics coverage report generated from the live
      writer, stage, and storage catalogs.
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
- [x] Record the fail-closed track adoption blocker: two structured lineage
      dtypes cannot yet round-trip through `DTypeContract`, `StoragePlan`, the
      array factory, and physical metadata comparison.
- [ ] Define exact Arrow dtypes for export columns where cross-language stability
      requires them.
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
- [ ] Record every requested component's output identity, exact dependency
      handle, and validation result in the chaser runner receipt. Individual
      writers now return a digest-bound receipt; orchestration adoption remains.
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
- [ ] Subject-shape candidate adoption.
- [ ] Track-kinematics candidate adoption.
- [x] Shared columnar adoption for swim bouts and bout kinematics. The new
      candidate publisher rematerializes exact compact runs through the shared
      byte planner without changing either production writer or selector.
- [x] Add the shared exact compact rematerialization boundary used by that
      adoption: it derives the growth axis from semantic axes, creates arrays
      through the common factory, writes complete physical units, preserves
      explicit report artifacts, and validates an executable receipt.
- [ ] Compact stimulus-response v3 candidate adoption.

Recommended migration order:

1. eye angles;
2. tail kinematics;
3. subject shape;
4. tail-posture view and bout classification;
5. track kinematics;
6. shared columnar storage, thereby covering swim bouts and bout kinematics;
7. stimulus response after its exact schema is frozen;
8. stimulus/occupancy/chaser families after their authority classification and
   lifecycle are complete.

### Phase 7 — Complete publication and registry projection

- [x] Add serialized registry completion/invalidation projection for all nine
      maintained catalog stages. The finalizer reopens each archive directly,
      requires matching `latest`/`latest_complete`, completion, and selector
      eligibility, and dispatches registry writes serially; no worker owns the
      SQLite transaction.
- [ ] Decide production selection policy for body frame and keypoint quality;
      do not infer activation from their selector-ineligible canaries.
- [ ] Standardize copy-integrity policy across maintained families.
- [ ] Require direct and consolidated metadata equivalence before visibility.
- [ ] Ensure completion, eligibility, selector activation, and registry state
      cannot disagree after recovery.
- [ ] Add idempotent finalizer and retry tests.

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

- [x] Add a deterministic suite planner that binds every array workload and
      the whole-run publication workload to the exact logical declaration,
      storage-plan receipt, logical dimensions, seed, and safety policy. It
      covers 200,000- and 1,000,000-row contract scales and rejects rehashed
      plan, selection, publication, or eligibility-policy tampering.
- [ ] Define a deterministic writer/publisher/reader workload for every
      maintained family.
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

Continue exact-contract and opt-in candidate work without promoting defaults:

1. finish the disjoint stimulus-response and chaser-component lanes;
2. keep shared catalog/planner/registry/selector edits in this serial
   coordination lane;
3. integrate each family lane serially and run the combined catalog, lifecycle,
   exact-schema, and storage-receipt matrix; and
4. run family-specific writer/rematerialization/publication/reader benchmarks
   for the integrated selector-ineligible swim-bout and bout-kinematics
   candidates at representative short and full-duration scales.
