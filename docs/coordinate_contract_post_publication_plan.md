# Coordinate contract post-publication plan

Status: proposed implementation handoff
Prepared: 2026-07-21
Execution prerequisite: publish and fast-forward the active coordinate-contract
remediation branch, then begin this work from a clean branch based on the exact
published commit.

## Purpose

Palette's coordinate semantics are sound and should remain authoritative.  The
next phase is not a redesign of the scientific model.  It is a targeted effort
to:

1. close the remaining normal-path coordinate/calibration coverage gaps;
2. verify the acquisition lens-distortion/rectification assumption;
3. distinguish durable construction history, current validity, and publication
   state;
4. reduce repeated process-local seal and exact-type ceremony without weakening
   scientific or crash-consistency guarantees; and
5. finish adopting shared publication and serialization machinery that already
   exists.

This plan supersedes recommendations to treat the current single-writer
compactors and finalizers as active incomplete-run exposure bugs.  Current run
resolution requires both completion and selector eligibility, and those writers
validate before completing.  Their lifecycle differs from the stricter
eligibility-last future-normal protocol, but that difference is contract and
concurrency-policy debt rather than evidence that readers can select an
incomplete run.

## Decisions already made

### Use Zarr format 3 only

Palette's active archives, writers, readers, registry discovery, and contract
audits target Zarr format 3. New code must use `zarr.json` and must not add
`.zgroup`, `.zarray`, `.zattrs`, `DirectoryStore`, or other Zarr-format-2
fallbacks. Encountering a format-2 archive is an unsupported-input error, not a
compatibility path.

Version suffixes on Palette schemas and scientific contracts are independent of
the Zarr storage format. Names such as `source_video_metadata.v2`,
`directed_transform_v2`, and pose-schema `v2` remain valid and must not be
removed by storage-format cleanup.

### Preserve the scientific contract

Keep all of the following:

- controlled coordinate spaces and typed profiles;
- array-owned coordinate or measurement semantics;
- explicit geometry type, components, units, axes, origin, extent, and pixel
  convention;
- distinct pixel-center, continuous-point, and half-open pixel-edge semantics;
- exact row identity, temporal mapping, and collection-axis identity;
- typed, directed transforms with exact source and destination authorities;
- explicit source-camera, ROI, model-input, canvas, physical, and anatomical
  frames;
- canonical future writers and isolated legacy compatibility paths; and
- refusal to combine arrays whose frames or acquisition-time domains do not
  agree.

Do not return to convention-based frame inference or group-level coordinate
labels.

### Preserve operational safety

Keep protections against realistic failures:

- immutable public run names;
- node-local staging and atomic installation where applicable;
- completion gating;
- fresh authoritative-path reopening after persistence;
- exact attempt ownership;
- conditional rollback that stops on ownership loss;
- selector generation/lease protection for concurrent publishers; and
- fail-closed resolution of malformed, incomplete, or selector-ineligible runs.

These address LSF concurrency, process death, stale handles, PRFS behavior, and
write operations that persist before raising.  They are not the subtraction
target.

### Simplify only process-local ceremony

Object-identity seals, exact-class rejection, and defenses against malicious
mapping subclasses may be simplified when current persisted-evidence validation
provides the same guarantee.  A seal may remain as a small process-local
capability indicating that an approved validator produced an object during the
current process.  It must not be treated as durable proof of how the underlying
data was created.

No seal or type gate should be removed until characterization tests show that
wrong frames, identities, transforms, paths, payloads, or archives are still
rejected.

## Evidence model

Treat four persisted claims separately:

| Claim | Question | Required evidence |
|---|---|---|
| Construction | How was this run produced? | Producer and code identity, parameters, input run/artifact refs and digests, model digest, output manifest digest, job/time metadata |
| Integrity | Is this still the same data? | Exact logical or physical payload manifest and digest policy |
| Validation | Does this exact output satisfy its contract? | Validator/ruleset identity, contract version, validated manifest digest, result and timestamp |
| Publication | Is this the selected approved result? | Completion state, owner/generation lease, selector transition, validation receipt used for activation |

An optional process-local `Verified[T]` capability answers only: "did the
approved loader/validator create this wrapper in this process?"  It does not
replace any persisted claim above.

Existing Palette records should be reused rather than duplicated.  Before
introducing new attributes, map the proposed claims onto current run provenance,
coordinate derivation records, array descriptors, payload digests, completion
metadata, selector leases, and atomic-publication metadata.  New schemas should
fill demonstrated gaps, not restate existing records under new names.

## Work package 0: establish the published baseline

After the other agent publishes the active branch:

1. Record the published Git commit and confirm the shared `/groups` checkout is
   fast-forwarded to it.
2. Start a new focused branch from that commit; do not perform this plan in an
   older agent worktree.
3. Confirm the worktree is clean and inventory other active branches touching
   coordinate publication modules.
4. Run the existing focused coordinate, publication, calibration, chaser, and
   stimulus-response tests outside the sandbox according to repository policy.
5. Retain the baseline test results and current serialized-record fixtures.

Exit criteria:

- one recorded base commit;
- clean focused branch;
- baseline suites green or every pre-existing failure documented; and
- no production Zarr or registry mutation.

## Completed foundation: bounded provenance-graph verification

Status: implemented and locally validated on 2026-07-21.

Canonical chaser publication previously re-opened the same selected-calibration
authority at nearly every edge of its nested provenance graph.  This was not
additional scientific validation: the calls proved the same sealed object,
archive identity, canonical paths, and digests thousands of times inside one
synchronous high-level operation.

The first optimization slice introduces an operation-scoped proof session with
these constraints:

- the cache is context-local, not process-global or time-based;
- nested calls join only the current outer operation;
- keys bind the validated object's identity, archive identity, canonical paths,
  schema version, manifest digest, matrix digest, and transform digest;
- an initial persisted verification is always required;
- every distinct proof is re-opened and verified again before the outer
  operation returns successfully;
- a later operation always starts with an empty session; and
- chaser activation's three deliberate publication checkpoints remain separate
  operations rather than sharing one cache across the commit point.

This assumes selected-calibration authorities remain immutable during one
operation.  A mutation present at the closing check fails the operation.  A
transient mutation that is introduced and fully restored between the initial
and closing checks is not observable, so this mechanism must not be extended to
mutable authorities without a stronger generation contract.

Same-machine local-Zarr A/B results using the canonical chaser fixture, with
memoization disabled in memory for the baseline, were:

| Phase | Baseline | Scoped verification | Full calibration checks |
|---|---:|---:|---:|
| Build | 5.035 s | 1.560 s | 684 -> 4 |
| Write, publish, and activate | 72.599 s | 16.853 s | 10,720 -> 12 |
| Total | 77.634 s | 18.414 s | 11,404 -> 16 |

The total path improved by about 4.2x.  The remaining time is real Zarr output,
descriptor/publication validation, and deliberately independent activation
checks rather than repeated selected-calibration graph traversal.

Validation evidence:

- six deterministic proof-session lifecycle tests passed;
- selected-calibration scope/fresh-call tests passed;
- the canonical chaser reader proves exactly one initial and one closing full
  calibration check; and
- all 114 focused proof-session, selected-calibration, chaser mutation, reader,
  publication, activation, and rollback tests passed in 172.49 seconds after
  the final object-identity key hardening.

Do not generalize this mechanically to every `assert_verified()` call.  Extend
it one immutable authority family at a time, with a stable proof key, an
explicit high-level operation boundary, mutation tests, and an A/B showing that
the eliminated work is genuinely duplicate.

### Stimulus-import extension

The second bounded slice applies the same session only at explicit stimulus
import call sites:

- coordinate-contract materialization;
- physical-coordinate authority publication;
- post-completion coordinate and physical-authority reloads; and
- each individual selector-activation proof invocation.

There is deliberately no scope around the whole import.  Completion remains a
freshness boundary, and the three activation proof calls each start with an
empty session.  A regression test mutates the selected camera immediately after
the activation lease write and proves that the next proof observes the drift,
fails the candidate, and leaves `stage_selector_eligible=false`.

The same-machine canonical stimulus fixture measured:

| Path | Wall time | Full calibration checks |
|---|---:|---:|
| Baseline | 49.734 s | 5,553 |
| Scoped implementation | 11.029 s | 26 |

This is about a 4.5x wall-time improvement.  Before the change, 5,339 of the
5,553 checks came from the same selected-canvas lineage construction and
validation pair; another 205 came from repeated transform-semantics checks.
After the change, the retained checks represent distinct materialization,
publication, reload, and activation boundaries.

All 73 stimulus-import context/path tests, including ownership takeover,
failure cleanup, final H5 revalidation, post-completion reload failure,
interrupt handling, source-file replacement, and the new activation-drift
case, passed in 187.01 seconds.

### Subject-shape publication extension

Subject-shape publication now treats proof reuse as three explicit phases:

1. validate the refined-mask source and newly written child while the child is
   running and selector-ineligible, then close the phase before completion;
2. freshly validate the completed child and close the phase before the first
   parent-selector mutation; and
3. after the guarded selector writes, restart from an empty proof set, reload
   the complete child and its exact source graph, and close that phase before
   the literal final `stage_selector_eligible=true` mutation.

Deferred activation performs its final proof in a separate operation and also
closes that proof before eligibility. Strict public subject-shape readers use
one bounded operation scope, so they deduplicate shared acquisition/import
authorities during a read while still reopening every distinct proof at the
closing check. No cache survives an operation or publication phase.

The motivating two-row cProfile traversed the provenance DAG as a fresh tree
from every descriptor edge: 1.96 billion Python calls, including 388,772
acquisition-frame validations, 777,604 import-ownership validations, and 3.2
million archive-identity resolutions. These were shared-root validation fanout,
not row processing or an equivalent number of filesystem reads.

Using the retained canonical two-row archive on the same workstation:

| Path | Unscoped baseline | Phase-scoped implementation |
|---|---:|---:|
| Write, publish, and activate | 150.639 s | 25.139 s |
| Strict completed-publication read | about 70 s in the baseline profile | 3.802 s |

The writer improved by about 6x while preserving selector rollback, interrupted
eligibility, deferred activation, source-tamper rejection, and the rule that
eligibility remains the final mutation. Canonical upstream test-fixture
construction remains a separate performance target and is not hidden by this
change.

The final warm-fixture cProfile recorded 44,549,213 calls, down from
1,957,304,450 (97.7% fewer, or about 44x). The high-fanout shared roots changed
as follows:

| Validation/root helper | Baseline calls | Final calls |
|---|---:|---:|
| Acquisition camera frame | 388,772 | 653 |
| Acquisition import ownership | 777,604 | 24 |
| Archive identity | 3,198,154 | 63,224 |

The final profile retains exactly three complete subject-shape publication
loads, corresponding to the deliberate completed-child, pre-selector, and
post-selector checkpoints. Remaining cumulative time is dominated by Zarr's
synchronous metadata path and real manifest/schema-inventory reconstruction.

The upstream canonical refined-mask test source is now a persistent,
content-addressed immutable fixture. Its first population took 5m50s and the
identical steady-state writer test took 33.94s. Cache hits verify the complete
directory digest plus essential Zarr v3 lifecycle, shape, and dtype metadata;
each test mutates only a private clone. Relevant source or NumPy/Zarr version
changes select a new cache key. The key hashes the exact source text of the five
fixture-construction helpers rather than the whole tamper-test module, so an
unrelated assertion edit does not discard the expensive source graph. Both the
writer/materializer and strict-reader modules share the same entry.

The full warm-cache subject-shape writer and materializer file passed all 16
tests in 8m12s. The single broad strict publication/tamper regression, which
performs about twenty real reloads, passed in 3m27s with the shared cache warm;
the ordinary coherent-writer regression passed from the other module against
the same entry in 34.93s.

### Observation and subject-mask publication extension

Status: implemented and locally validated on 2026-07-23.

Shard 8 exposed the same shared-root validation fanout in observation geometry
and raw/refined subject-mask publication. The motivating two-row profiles were
not processing meaningful image volume:

- one refined-mask activation test made 779,684,374 calls, including about
  680,000 persisted-proof checks, 208,000 acquisition-frame validations, and
  416,000 import-ownership validations; and
- one crop-ROI publication test made 814,859,186 calls, with about 272 of 274
  profiled seconds below persisted-proof verification.

The extension gives each standalone high-level observation publication/load
call a fresh operation scope. Raw and refined subject-mask prepare, publish,
strict-load, completed-ineligible-load, and activation calls now do the same.
Nested calls still join an owning writer's explicit scope. In particular, the
refined finalizer retains its existing combined prepare/publication phase and
closes it before completion. Raw subject-mask activation now closes and freshly
rechecks its completed-child proof before acquiring the parent lease or writing
selectors. No proof survives an operation, lifecycle transition, or activation
commit point.

Proof-heavy fixture construction is also explicitly scoped in the observation
and directed-transform tests. This removes redundant setup validation while
leaving every tested publisher, loader, mutation, and rollback call in its own
fresh operation. It is not a persistent fixture cache and does not share trust
between tests.

Same-workstation results were:

| Path | Unscoped/previous | Scoped implementation |
|---|---:|---:|
| Representative crop-ROI + refined activation pair | 227.25 s | 20.17 s |
| Complete observation publication module | 87.79 s before fixture scoping | 18.55 s |
| Refined publication + proof-session lifecycle modules | historical slowest tests alone exceeded 13m | 16.52 s for all 38 tests |
| CI shard 8 pytest body | 29m17s | 4m26s |

The exact shard-8 assignment still contained the directed-transform,
observation-publication, and refined-mask-publication modules; small supporting
file membership changed when the deterministic source-weight sharder was
recomputed. The end-to-end result was 528 passed, one skipped, and 14 expected
failures in 266.94 seconds. The focused observation/raw/refined publication run
passed all 104 tests, and the directed-transform suite passed all 96 tests.

The retained regression coverage includes same-shape payload tampering,
descriptor and authority swaps, a fresh activation payload scan, concurrent
selector and publication-epoch mutation, interrupted activation rollback,
alien parent-state preservation, and the rule that child eligibility remains
the final commit mutation.

## Work package 1: close normal-path calibration coverage

This is the first scientific implementation priority.

### Stimulus response and OMR

Status: implemented and focused validation green on 2026-07-21, with the
angular-direction and imaging-model qualifications below still open.

Normal stimulus-response computation still reaches historical calibration
resolution in:

- `src/fisheye/analysis/stimulus_response.py`;
- `src/fisheye/analysis/stimulus_response_omr.py`; and
- any helper reached from those modules that probes global/root calibration or
  applies a transform without typed direction and frame checks.

Required changes:

1. Load the exact selected stimulus/calibration snapshot and typed frame
   authorities before computing loom, grating, concentric, or arena geometry.
2. Resolve physical scale and centers from that bound evidence rather than
   scanning multiple calibration locations.
3. Apply only direction-labelled transforms whose source/destination frames and
   extents match the numeric inputs.
4. Persist the exact input authority and derivation references with the output
   metrics.
5. Keep historical fallback behind an explicit legacy-compatibility entry point;
   it must not silently feed a normal selector-eligible run.
6. Compare old and new numeric outputs when they resolve the same underlying
   calibration.  Any discrepancy must be explained before publication; do not
   relabel changed values as a metadata-only migration.

The normal path now loads a `StimulusResponseCoordinateAuthority` from the
selected, complete, selector-eligible stimulus run.  The authority binds the
canonical stimulus coordinate-output manifest, arena frame, selected canvas,
source-camera frame, directed arena-to-camera transform chain, selected
calibration, and source-camera physical scale.  It also requires the selected
track-motion run to use the exact same physical-authority identity.  The
low-level writer repeats that cross-input check, so callers cannot bypass it by
invoking publication directly.

Loom, concentric, and moving-grating arena geometry no longer scan
`analysis/calibration`, root `calibration`, or partially populated per-run
attrs.  Arena points are transformed through the typed
arena-to-selected-canvas-to-source-camera chain and then scaled with the exact
camera millimetres-per-pixel authority.  Explicit precomputed millimetre
coordinates are accepted only when labelled `source_camera_physical_mm`.
Non-centre loom placement remains fail-closed until Citrus materializes a typed
placement surface.

This is a justified recomputation rather than a metadata-only migration.  The
selected-calibration matrix is declared camera-to-canvas; historical response
helpers applied a direction-neutral matrix as though it were canvas-to-camera.
In the canonical numeric fixture, canvas point `(442, 692)` maps through the
explicit inverse to camera point `(432, 672)`, or `(8.64, 13.44)` mm at 50
camera pixels/mm.  Applying the same stored matrix forward would instead yield
`(9.04, 14.24)` mm.  The real-Zarr regression test fixes this direction and
also rejects a physically equivalent-looking authority copied from another
stimulus run because its durable record identities differ.

New publications use `stimulus_response.v3` and persist a digest-bound
`source_refs.stimulus_coordinate_lineage`.  The production materializer
validates that lineage before accepting its output.  Focused evidence is 127
stimulus-response writer/materializer tests plus two canonical imported-Zarr
authority tests passing outside the sandbox.

The remaining moving-grating angular offset is not silently certified by this
change.  Existing per-step `direction_mapping_status`,
`direction_mapping_source`, and `direction_mapping_validated` metadata remain
authoritative; results whose mapping is unvalidated retain that status.  The
separate acquisition-direction calibration work must establish absolute
angular accuracy.  Work package 2 also remains open: the selected source frame
currently declares `image_space="raw"`, and no rectification or distortion
model has yet been demonstrated.

### Chaser distance

Do not reopen the already corrected normal chaser-distance path as a coverage
bug.  Its default builder uses sealed detection/stimulus coordinate evidence.
The historical calculation remains only for explicit legacy compatibility.
Maintain tests proving that default execution cannot enter that legacy path.

Exit criteria:

- every normal stimulus-response spatial/physical calculation starts from typed
  selected evidence;
- normal writers fail closed when evidence is absent or inconsistent;
- explicit legacy behavior remains readable but cannot become canonical; and
- numeric parity or justified recomputation is documented.

## Work package 2: verify the camera imaging model

The current selected-calibration authority binds a 3x3 camera-to-canvas
homography.  A homography models planar projective geometry but cannot generally
represent radial or tangential lens distortion.  Palette currently does not
declare whether `source_camera_image_px` is raw/distorted or rectified.

Perform a read-only acquisition/calibration audit before adding new transform
types:

1. Identify the camera and lens models used by the active cohorts.
2. Trace whether Citrus, camera firmware, acquisition, transfer, or import
   rectifies frames before Palette sees them.
3. Inspect H5/calibration artifacts for intrinsic matrices, distortion
   coefficients, rectification maps, model names, or calibration residuals.
4. Establish which pixel surface was used to estimate each homography.
5. Measure or recover calibration residuals across the full arena, preferably
   in both pixels and millimetres and with held-out calibration points.

Choose one documented outcome:

- **Rectified upstream:** persist a digest-bound rectification/imaging-model
  reference and declare the source frame rectified.
- **Raw but negligible at the required scale:** record the lens model,
  validation method, spatial coverage, residual distribution, and an approved
  error threshold.
- **Material nonlinear distortion:** introduce a new typed transform kind for
  distortion/rectification or a persisted lookup map.  Do not certify it as a
  homography.

The repository name `fisheye` is not evidence that the physical lens is a
fisheye lens.  Make this decision from acquisition records and measured error.

Exit criteria:

- distortion state is known for each supported acquisition family;
- the homography assumption has quantitative evidence or is explicitly
  rejected; and
- any new transform work has a separate reviewed design before implementation.

## Work package 3: specify durable receipts without duplicating provenance

Write a small versioned design and mapping before changing persisted schemas.

1. Inventory which existing fields already satisfy construction, integrity,
   validation, and publication claims.
2. Define a construction receipt that binds producer/version, Git or package
   identity, parameters, input refs/digests, model/artifact identity, and the
   output manifest.
3. Define a validation receipt bound to an exact output-manifest digest and a
   versioned validator/ruleset identity.
4. Define a publication receipt or normalized view over existing completion,
   owner, generation, selector, and atomic-publication records.
5. Decide whether receipts are new stored records or typed views over existing
   records.  Prefer views and references when they avoid duplicate authority.
6. Define invalidation:
   - immutable runs retain receipts while their manifest matches;
   - edits advance a revision/generation and invalidate prior validation;
   - compaction creates a new construction receipt for the new immutable run.
7. Treat cryptographic signatures as optional.  SHA-256 supplies integrity
   binding, not authorship, when an actor can modify both payload and digest.
   Add signing only if tamper-authenticated chain of custody becomes a real
   requirement.

Exit criteria:

- no duplicated source of truth;
- each claim has one documented authority;
- receipts bind exact immutable content or an explicit edit generation; and
- readers can explain why a run is constructed, intact, valid, and selected as
  four separate answers.

## Work package 4: pilot a hybrid verified-object model

Use one small leaf authority family, initially
`src/fisheye/shared/track_coordinate_publication.py`.  Do not begin with core
identity, descriptor, calibration, or mask modules.

### Hypothesis

A generic validated dataclass plus fresh persisted-evidence validation can
replace family-specific object-identity seals and exact-type checks without
weakening scientific or operational behavior.

### Pilot changes

1. Freeze current successful serialized records, digests, and error behavior in
   characterization tests.
2. Introduce or reuse one generic process-local `Verified[T]`/validation-receipt
   mechanism rather than another family-specific seal.
3. Replace `type(value) is BoundType` with validated protocol/dataclass checks
   only where subclass identity is scientifically irrelevant.
4. Permit read-side `Mapping` inputs only after copying them into exact built-in
   JSON values and running strict canonical/schema validation.
5. Keep write-side Zarr attributes behind a controlled transaction adapter; do
   not allow arbitrary mappings to become mutation targets.
6. Remove repeated validation only when no persistence, process, ownership,
   selector, or mutability boundary has been crossed.
7. Preserve fresh authoritative-path reload and comparison at every durable
   boundary.

### Required hostile cases

- wrong archive or canonical path;
- changed record or payload digest;
- wrong row identity or temporal mapping;
- incompatible frame or transform direction;
- stale object after backing-store mutation;
- partial attribute write;
- persistence followed by an exception;
- rollback ownership loss; and
- deliberately direct construction of a plausible but unvalidated dataclass.

### Evaluation

Measure:

- production and test LOC removed;
- number of family-specific constructors/gates eliminated;
- serialized byte/digest parity;
- failure-case parity;
- readability of producer and consumer boundaries; and
- whether any test became less precise.

Stop or redesign the pilot if direct construction can cross a durable boundary
without fresh validation, or if any scientific mismatch previously rejected is
accepted.

## Work package 5: finish adopting existing shared infrastructure

This is consolidation, not greenfield framework construction.

1. Inventory private canonical JSON, SHA/digest, attribute snapshot/restore,
   and rollback helpers by their persisted grammar and failure semantics.  The
   current repository has dozens of local implementations; do not assume they
   are equivalent merely because their names are similar.
2. For each equivalent family, select the existing shared implementation as
   authority, add parity fixtures, migrate callers, and delete the private
   copies in the same work package.  Do not add a shared helper while leaving
   indefinite duplicate implementations behind.
3. Prefer the existing strict JSON/canonicalization helpers over private
   per-module implementations.
4. Prefer `selector_activation.py` for owner/generation-leased publication when
   the run family permits concurrent publishers.
5. Prefer `atomic_run_publisher.py` for node-local physical installation and
   failure handling.
6. Extract a shared physical-tree inventory/checksum helper only where current
   implementations are demonstrably equivalent.
7. Add shared publication conformance fixtures so stages supply proof loaders
   and scientific validation rather than cloning fault-injection tests.
8. Keep scientific source selection, coordinate derivation, component meaning,
   calibration choice, and numeric validation stage-specific.

Before modifying current completion-gated single-writer producers, make an
explicit policy decision:

- retain and document a versioned single-writer protocol whose selection gate
  is `complete AND eligible`; or
- migrate the family to the leased eligibility-last protocol for uniformity or
  future concurrency.

Do not describe the former as an incomplete-run exposure bug.  If retained,
document and test its single-writer assumption.  If simultaneous publications
are possible, migrate it deliberately to leased activation.

Exit criteria:

- one implementation of each adopted mechanical behavior;
- stage adapters retain explicit scientific proof loaders;
- no selector or serialized-schema behavior changes accidentally; and
- lifecycle differences are named policies rather than incidental code.

## Work package 6: extract the stage-neutral inference crop authority

`BoundKeypointCropSource` is already consumed by subject-mask publication.  Its
core semantics are therefore crop-backed inference semantics, not pose-only
semantics.

1. Introduce `BoundInferenceCropSource` in a neutral shared module.
2. Preserve `BoundKeypointCropSource` and existing loader names as compatibility
   aliases during migration.
3. Move only shared crop identity, subset/reorder validation, placement, ROI and
   source-camera frames, and model-input transform binding.
4. Keep pose schema, keypoint labels, mask components, probability thresholds,
   refinement authority, and measurement semantics in their respective stages.
5. Allow refined-mask tooling to consume the neutral crop evidence where
   appropriate, but do not make it the sole authority for refinement-specific
   lineage.

Exit criteria:

- masks no longer depend on a keypoint-named authority for their crop source;
- compatibility imports remain green; and
- no stage-specific semantics leak into the neutral type.

## Work package 7: remove coordinate-foundation dependency cycles

The current coordinate foundation contains a real strongly connected component:

```text
coordinate_descriptor
  -> coordinate_identity
  -> pixel_frame_authority
  -> selected_calibration
  -> directed_transform
  -> coordinate_descriptor
```

There is also a direct cycle between canonical coordinate publication and
coordinate-frame records.  Lazy imports currently prevent import-time failure,
but they make dependency direction difficult to understand and allow leaf
utilities to accumulate inside peer authority modules.

This is architectural debt, not an active scientific defect.  Address it after
the coverage, receipt, and verified-object work packages establish the desired
leaf primitives.

1. Generate and retain a source-level dependency graph and characterize every
   lazy edge before moving code.
2. Extract only genuinely leaf-level primitives first: canonical JSON/schema
   helpers, record refs, array-payload identity, and generic attribute
   transactions.
3. Separate pure record schemas from persisted authority loaders and from
   stage-specific adapters.
4. Move acquisition materialization, selected-calibration evidence, generic
   pixel-frame records, crop placement, physical/body frames, and publication
   adapters into a one-direction dependency layout.
5. Keep the current public module names as compatibility facades and re-export
   existing APIs while callers migrate.
6. Require serialized-record, digest, exception, and import-behavior parity.
7. Add a dependency check that prevents the removed strongly connected
   component from being reintroduced.

Target dependency direction:

```text
JSON and record primitives
  -> pure schemas
  -> persisted authorities
  -> generic descriptor/publication machinery
  -> stage-specific profiles and producers
```

Exit criteria:

- no load-bearing lazy imports within the former coordinate-foundation cycle;
- public compatibility imports remain green;
- no persisted schema or numeric behavior changes; and
- the dependency-direction check runs in CI.

## Work package 8: consolidate documentation authority

The repository currently has hundreds of Markdown files, including many dated
diagnostics and implementation handoffs.  Dated evidence is useful, but it must
not compete with stable contracts or force a reader to reconstruct the current
decision from a timeline of agent snapshots.

1. Inventory every non-archived document and classify it as:
   - authoritative contract;
   - active implementation plan;
   - operational runbook;
   - retained diagnostic evidence;
   - superseded handoff/status snapshot; or
   - obsolete/redundant.
2. Give each subject one indexed authoritative entry point.  That entry point
   must name the current contract/decision and link retained historical evidence
   without copying its conclusions.
3. Merge repeated decisions and checklists into the authoritative document.
4. Move superseded dated snapshots to `docs/archive/` when they retain useful
   evidence; delete them only when their evidence and decisions are completely
   duplicated and no audit reference depends on them.
5. Keep diagnostic artifacts as evidence, but clearly label them non-authority
   and link them from the corresponding stable decision.
6. Add lightweight document metadata or an index that exposes status,
   authority, successor, and last verification date.
7. Make every new dated handoff name the stable document it updates or the
   explicit evidence gap it fills.

Exit criteria:

- a new engineer can find the current coordinate, provenance, and publication
  decisions without reading dated snapshots in chronological order;
- no two active documents claim conflicting authority over the same contract;
- superseded documents have an explicit successor or archive status; and
- document count decreases through reviewed consolidation rather than bulk
  deletion.

## Work package 9: defer declarative surface compilation

A typed `CoordinateSurfaceSpec`/`MeasurementSurfaceSpec` compiler could reduce
substantial repeated binding code, but it touches many actively developed
publication modules.  Defer it until the current branch/worktree wave has
settled and the smaller pilots above establish the required primitives.

When begun:

1. keep existing public modules as compatibility facades;
2. canary one relatively small surface family first, preferably tail;
3. preserve exact serialized records and error codes;
4. migrate one family per commit;
5. retain stage-owned specs and evidence resolvers; and
6. do not create a generic catch-all provenance dictionary.

## Validation and deployment policy

For every work package:

- start with deterministic in-memory characterization tests;
- preserve exact serialized records unless an intentional schema version is
  introduced;
- run Zarr-heavy focused tests outside the sandbox using `scripts/py`;
- use fault injection for partial writes, persisted-then-raised writes, stale
  handles, takeover, and rollback;
- validate bounded/streaming payload verification for large arrays rather than
  introducing unconditional full-raster rereads;
- run a non-promoted LSF canary before changing active selector behavior; and
- publish/promote only after fresh final-path validation.

Production data migration is not implicit authorization.  Any backfill or
recomputation must have its own dry-run inventory, value-validity decision,
reviewed mutation plan, and rollback/immutability policy.

## Suggested commit boundaries

Keep these reviewable and independently reversible:

1. documentation and baseline inventory;
2. stimulus-response typed-calibration coverage;
3. imaging-model audit and decision record;
4. receipt model and contract tests;
5. one leaf verified-object pilot;
6. shared-substrate adoption by one publisher family;
7. neutral inference-crop extraction;
8. coordinate-foundation dependency-cycle removal;
9. documentation authority consolidation; and
10. later one-family-at-a-time declarative surface migration.

Do not combine scientific numeric changes, persisted-schema changes, lifecycle
changes, and broad file movement in one commit.

## Explicit non-goals

- removing coordinate descriptors or typed frame authorities;
- replacing array-owned semantics with path/name conventions;
- deleting every seal mechanically;
- weakening fresh-store validation or crash-consistency safeguards;
- changing existing production selectors as part of refactoring;
- rewriting historical Zarrs without a separate migration decision;
- implementing nonlinear lens correction before verifying the imaging model;
- deleting dated documentation without classifying its authority and retained
  evidence;
- splitting every large module solely to reduce line counts; or
- launching a monolithic rewrite across all coordinate publication families.
