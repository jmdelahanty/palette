# Source-of-Truth Consolidation Review and 10-Step Plan

**Date:** 2026-08-25

**Method:** eight parallel read-only Luna xhigh reviews of recording identity,
frame authority, epochs and speed, run selection and completion, mirrors and
manifest coverage, registry reconciliation, omitted authority paths, and the
cross-cutting architecture, followed by direct source confirmation.

**Repo state:** branch
`agent/palette/clipped-geometry-acquisition-authority-20260821`, HEAD
`e58443c3` (`docs: record track reader optimization audit`).

**Status:** design and sequencing record. This review changes no production
authority, selector, registry, or reader implementation.

**Companion audits:**
[`redundancy_campaign_2026-08-24.md`](redundancy_campaign_2026-08-24.md),
[`pipeline_survey_2026-08-24.md`](pipeline_survey_2026-08-24.md),
[`track_motion_reader_optimization_2026-08-24.md`](track_motion_reader_optimization_2026-08-24.md),
[`crop_contract_split_audit_2026-08-24.md`](crop_contract_split_audit_2026-08-24.md),
[`contract_enforcement_divergence_review_2026-08-21.md`](contract_enforcement_divergence_review_2026-08-21.md),
and [`subtraction_queue_2026-08-21.md`](subtraction_queue_2026-08-21.md).

---

## 0. Verdict

The redundancy campaign has the right failure model: the largest safety risk is
not rigor, but independently evolving implementations of the same claim. The
campaign needs a more precise rule than either "one physical copy" or "one
validation call":

> **One canonical authority per semantic fact. Every other published or
> consumed copy must be a digest- or revision-bound projection, an
> equality-checked compatibility mirror, or an explicitly non-authoritative
> cache or receipt.**

Independent evidence inputs may coexist to establish or challenge that
authority, but evidence must remain identified as evidence and cannot silently
become another precedence-based authority.

One resolver may dispatch across multiple supported publication profiles, but
each profile must receive full-strength validation. A receipt may replace a
rescan only when it proves the same claim for the same immutable artifact, its
declared phase and claim remain applicable, and identity, path, owner, manifest,
selector, eligibility, and generation changes since minting are checked
independently.

This changes the implementation priority from the pipeline survey's narrow
receipt-first sequence. The first two projects should be recording identity and
typed frame counts. Those are the clearest Tier 1 areas where live copies are
written or read under opposing precedence without an effective conflict gate.
Receipt-backed performance work remains important, but it depends on cleaner
metadata, run-resolution, lifecycle, and digest boundaries.

The intended shape is:

```text
observed evidence from manifests, Zarr, probes, and ledgers
                              |
                              v
           one typed, fail-closed resolver per semantic fact
                              |
              +---------------+----------------+
              |               |                |
       effective fact   bound projection   exact mirror
              |                                |
              +-------> registry/index <-------+

cache and operational telemetry remain outside the authority path
```

## 1. Terms and authority taxonomy

Every duplicate identified by the campaign should receive one of these
dispositions before code is removed or a new contract is added.

| Kind | Meaning | Required behavior |
|---|---|---|
| Canonical authority | The accepted source for one precisely named semantic fact. | One owner and one validation/resolution implementation. Conflicts fail closed. |
| Evidence | An independently observed input used to establish or challenge authority. | Preserve source, locator, digest or revision, and observation method. Do not silently apply precedence on conflict. |
| Projection | A reproducible copy created for a bounded consumer or index. | Bind to the authority identity and digest/revision; validate the binding on load or rebuild. |
| Mirror | The same fact written in two locations for compatibility or atomic visibility. | One writer and exact equality comparison on every authoritative read. |
| Cache | Recomputable convenience data. | Explicitly non-authoritative; safe to invalidate or delete; never selected as a fallback authority. |
| Receipt | Evidence that an operation or validator accepted a particular immutable object. | Closed schema, self-digest, claim scope, artifact identity, lifecycle phase, validator identity, and manifest binding. It grants only its declared guarantee. |
| Lifecycle or selector state | Mutable publication state such as completion, eligibility, and active selection. | Kept outside the immutable scientific digest/claim, even when physically colocated, and checked against the selected manifest, owner, and generation. |
| Supported profile | A distinct, intentional publication grammar serving a real workflow. | Reachable through the shared resolver, fully validated, and covered by a real-writer-to-unpatched-reader test. |

This taxonomy produces five immediate rules:

1. Names such as `total_frames`, `latest`, `status`, and `speed` are not
   sufficiently typed to be authorities by themselves.
2. A fallback ladder is not conflict handling. When two non-null observations
   disagree, the resolver must report or reject the conflict rather than hide
   it behind precedence.
3. Legitimate write-then-reopen verification remains. Validation at a new trust
   boundary is not redundant merely because the same validator ran earlier.
4. Registry rows remain projections for scientific content and authorities only
   for the registry-owned identity, locator, and minted-entity facts declared in
   [`registry_data_governance_policy.md`](../registry_data_governance_policy.md).
5. Deletion follows migration evidence. A compatibility path is removed only
   after its callers are migrated or its profile is explicitly tombstoned.

### 1.1 Vertical versus horizontal checks

The distinction is a useful heuristic:

- **Vertical checks** usually cross a trust boundary: source to derived
  artifact, candidate to publication, direct to consolidated visibility, or
  published bytes to consumer. Manifests, content digests, reopen validation,
  atomic publication, owner/generation checks, and fail-closed readers are
  necessary on multi-object Zarr storage over a distributed filesystem.
- **Horizontal checks** often compare representations of an apparent peer fact:
  root versus `raw_video`, manifest versus loose attrs, registry versus Zarr, or
  two meanings of `latest`. When both sides became independent authorities by
  accident, the right fix is one authority and deletion of the redundant copy
  or writer.

It is not a keep/delete rule by itself. Some horizontal checks are load-bearing:
an intentionally rebuildable registry projection needs bounded reconciliation;
direct and consolidated metadata views must agree at publication; and a
compatibility mirror needs exact comparison while both locations remain in
use. Conversely, a vertical full-tree rescan is redundant when it repeats the
same claim over the same immutable object and lifecycle phase already covered
by a bound receipt.

The Iceberg/Delta analogy is directionally useful but should not be overstated.
A checksum does not by itself provide a transaction. Palette's effective commit
protocol also needs immutable staging, a closed manifest, owner/generation or
lease checks, an atomic selection transition, lifecycle validation, and a
validated consolidated metadata generation. The guarantee comes from that
whole protocol, not from the presence of SHA-256 fields.

Agent-authored code increases the value of centralized forcing functions,
types, mutation tests, CI, and runtime checks at untrusted boundaries. It is not
a reason to retain duplicated validators or add ambient checks at every call
site.

Judge each check by the claim and boundary it closes. Retain it only when it:

1. names its authority, artifact/profile, lifecycle phase, and failure model;
2. runs on a real consumer/publication path or on a bounded scheduled audit;
3. rejects a mutation or failure injection that would otherwise reach a
   consumer;
4. is not already proven by construction or by independently bound evidence;
   and
5. does not merely choose precedence between unresolved competing writers.

Therefore raw check count is a useful warning signal, not the primary success
metric. The campaign should reduce duplicate check *implementations* and
repeated scans while preserving or increasing trust-boundary coverage. Track:

- semantic facts with more than one untyped writer or authority;
- production writes that bypass the canonical writer;
- ambient fallback ladders and unclassified mirrors/projections;
- distinct implementations per declared digest algorithm;
- repeated full scans per publication and per normal read;
- high-risk mutation cases that survive the ordinary consumer path; and
- asynchronous divergence detection windows.

The proposed "two-thirds engineering, one-third band-aid" split is not yet an
evidenced number. Step 1's disposition census and mutation matrix should produce
the defensible split.

## 2. Corrections to the source audits

The broader review confirmed the campaign's direction but found several factual
or architectural corrections that must be carried into implementation.

| Prior wording | Corrected finding |
|---|---|
| The ordinary `recordings` upsert freezes an existing `session_uuid`. | `COALESCE(excluded.session_uuid, recordings.session_uuid)` at `registry/db.py:2618` lets any non-null incoming value overwrite the existing value; it preserves the existing value only when the incoming value is null. The maintenance writer at `registry/maintenance.py:914` hard-overwrites and can also erase with null. Both are last-writer-sensitive. |
| The identity problem is confined to `recordings`. | `datasets` is also mutable in conflict with policy: `session_uuid=excluded.session_uuid` and a non-null incoming `recording_id` wins at `registry/db.py:2534-2536`. Governance declares `datasets.session_uuid` immutable generally and `datasets.recording_id` immutable for `artifact_kind='source_recording'`. Joined views can therefore consume identity from different registry copies. |
| There are two central frame-count resolvers. | There are at least three: `shared/frame_domains.py:492-503`, `analysis/stimulus_epoch_runs.py:132-145`, and `shared/metadata.py`'s compatibility resolver. The first two already apply opposite source-versus-stored precedence. |
| The five manifest-validator tables contain 57 direct production sites. | The location cells contain 54 direct calls or callback references: 16 subject-mask, 11 refined-detection, 10 crop, 8 canonical-detection, and 9 keypoint sites. Transitive validations are a separate call-graph question. |
| Five manifest families exclude `cluster_output_staging`. | At least two additional exclusion sites exist in `shared/zarr/subject_mask_core_publication.py:109` and `shared/tail_coordinate_publication.py:169`. A separate subject-mask batch writer also emits a different staging schema at `utils/run_subject_mask_batch_pipeline.py:1132,1178`. |
| `cluster_output_staging.parent_attrs_after` is write-only and can be deleted. | Candidate validators read it, including the subject-shape, track, tail, chaser-distance, and eye-angle benchmark paths. It is a phase-specific pre/post-publication snapshot whose name and semantics need clarification, not a pure deletion. |
| Registry/Zarr divergence detectors are entirely passive or absent. | `utils/check_recording_steps.py:3259-3422` has a `--status-source compare` path. It is useful but limited, manual, and unscheduled; it does not close the general reconciliation gap. |
| Track's physical-payload root is never re-verified. | Lightweight checkpoints skip it, but Track performs fresh physical verification before activation at `analysis/track_kinematics.py:2959` and again at `:12455`. The narrower gap is the absence of an ordinary post-publication audit consumer. |
| Sibling materializers can adopt Track's receipt in a few lines. | Subject shape, bout kinematics, and eye angle retain different summaries and, in one case, a different hash grammar. Each needs an explicit adapter or full-report retention, a scientific-manifest binding, and a real consumer before its receipt is an active guarantee. |
| Sealing the current `cluster_output_staging` record is sufficient to replace rescans. | The record is mutable across publication phases and mixes copy evidence, timestamps, parent snapshots, pointer-adjacent state, and final validation. It must not be bound wholesale into an already-sealed scientific manifest. Stable publication evidence and mutable operational telemetry need separate records. |

These corrections do not weaken the consolidation case. They narrow each
change to the exact duplicated guarantee and prevent cleanup from deleting a
real trust-boundary check or turning mutable telemetry into scientific truth.

## 3. Verdict on the ten ranked source-of-truth findings

| # | Finding | Review verdict | Consolidation direction |
|---:|---|---|---|
| 1 | Recording identity in the registry | **Strong agreement; broader scope.** The normal registration and maintenance paths observe different sources and both can overwrite identity. `datasets`, profiles, and joined views also carry identity. | Separate `recording_id` from `session_uuid`; add one evidence resolver, one registry projection writer, and an explicit versioned correction path. |
| 2 | Frame counts | **Strong agreement; model domains rather than choosing one precedence.** Source, acquisition, stored, run, and crop counts can legitimately differ. | One typed accessor requiring a requested domain and returning evidence/provenance. Unqualified count reads become compatibility-only. |
| 3 | Acquisition authority versus plain attrs | **Strong agreement; fix before reader migration.** The preflight can seed source metadata from an attr that is stored/clip-domain, after which the authority verifies the same seeded blob. | Guard seeding with independent source metadata, probe, or acquisition-clock evidence; compare source and stored domains explicitly. |
| 4 | Epoch windows | **Agreement with qualification.** Copies can be legitimate projections; path and run name alone are insufficient bindings, and swallowed errors are unsafe. | Make the canonical stimulus-epoch run authoritative. Bind copies by manifest/selection digest and reject missing or stale bindings instead of returning zero windows. |
| 5 | Speed | **Agreement with semantic separation.** Verified track-motion speed and raw centroid-difference speed are different measurements. | Make verified track motion the default physical-speed authority. Retain raw speed only under an explicit `speed_source`/measurement label and a versioned threshold/noise-floor policy for that measurement product. |
| 6 | "Which run is current" | **Strong structural agreement.** The problem is not that all modes are invalid; it is that callers use untyped ambient meanings and registry fallback can choose a run stricter selectors reject. | One resolver interface with closed modes such as authoritative, latest complete, inventory latest, pending, and source match. Return the selected mode and evidence. |
| 7 | Root/raw-video mirrors | **Agreement for exact semantic mirrors only.** Counts with different domains must not be equality-compared. | Exact-compare codec, colorimetry, encoder, source identity, and publication-status mirrors, or remove one copy. Domain-stamp non-equivalent counts. Route `native_detection_authority` through the publication-status comparator. |
| 8 | Group attrs versus manifest payloads | **Agreement, but blanket digest parity is unsafe.** Scientific metadata, lifecycle state, receipts, and telemetry have different mutation rules. | Version manifest schemas so immutable scientific metadata is sealed; validate lifecycle state against the manifest separately; keep telemetry out of scientific digests. |
| 9 | Detection completeness | **Strong agreement.** Run attrs, manifests, selectors, receipts, LSF JSON, and registry status are evidence layers with no single closing transaction. | Add one serial detection finalizer that validates the selected immutable publication and writes completion/registry projections in one controlled flow. |
| 10 | Registry versus Zarr overall | **Agreement with a policy boundary.** Zarr remains authoritative for science; the registry owns identity/locators. Existing comparison is manual and incomplete. | First schedule a read-only comparison with durable reports and bounded staleness. Do not introduce blind scheduled repair until the resolver and projection writers are proven. |

## 4. First implementation target: recording identity

### 4.1 Why precedence cleanup is insufficient

`resolve_dataset_id()` prefers existing Zarr identity attrs and consults the
manifest only when they are absent (`registry/db.py:606-612`). Normal recording
context extraction reads root attrs, embedded context, and the manifest in that
order (`registry/db.py:730-761`). The maintenance backfill reads the manifest
without opening the Zarr (`registry/maintenance.py:824-948`).

The fallback also treats `recording_manifest.recording_id` as a candidate
`session_uuid` when the latter is absent (`registry/db.py:609-610`). The new
resolver must preserve those as distinct facts and reject that substitution
unless a versioned manifest contract explicitly proves equivalence.

That disagreement cannot be solved safely by declaring either source globally
first. Zarr identity attrs are an import-time snapshot, while a manifest may be
legitimately corrected later. A correction should not silently rewrite the
import-time artifact evidence, and an old artifact snapshot should not silently
undo an approved correction.

The target semantics are:

- `recording_id` is the registry recording-entity key. It is often derived from
  recording context or a directory name, but is not inherently a path key. A
  change is an explicit alias/canonicalization repair, never a routine upsert of
  a primary key.
- `session_uuid` is acquisition-surface identity. It may participate in
  matching, but it is not a sufficient registry primary key and must not be
  silently changed.
- Zarr root identity attrs should be treated as import-time artifact evidence,
  not a mutable correction ledger. Current paths tend to freeze them, but they
  are not cryptographically or structurally immutable merely because they are
  attrs.
- `recordings`, `datasets`, profile tables, and status views are registry
  projections over one effective identity decision.
- Corrections are append-only revisions with actor, reason, evidence, prior
  revision, and compare-and-swap protection. The acquisition-batch assignment
  pattern in
  [`acquisition_batch_registry_contract.md`](../acquisition_batch_registry_contract.md)
  is the in-repo model.

### 4.2 Proposed resolver shape

Introduce a read-only `RecordingIdentityEvidence` result before changing any
writer. Its exact implementation name is open, but it should return:

```text
effective_recording_id
effective_session_uuid
effective_revision_or_digest
observations[]:
  source_kind
  source_locator
  recording_id
  session_uuid
  source_digest_or_revision
conflicts[]
resolution_status
```

The resolver should observe the capture manifest, Zarr artifact snapshot,
current registry rows, and any approved correction revision. An absent value
may be filled from compatible evidence. Two conflicting non-null values must
produce a conflict, not a precedence winner.

### 4.3 One projection writer

Replace the maintenance raw SQL and normal registration divergence with one
writer that:

1. consumes only a successfully resolved evidence result;
2. creates a missing projection or fills an unambiguous null;
3. rejects conflicting non-null identity unless an explicit correction
   revision authorizes it;
4. applies the same rules to `recordings` and `datasets`;
5. never changes a primary identity through ordinary rescan/backfill;
6. does not rewrite `dataset_id` or change path-hash identity as a side effect
   of resolving recording/session identity;
7. records the authority revision/digest and projection timestamp; and
8. emits a durable conflict report rather than silently choosing a source.

Profile and provenance tables should preserve and extend the existing
normalized-view pattern (`dataset_context_current.recording_id`) rather than
acting as another identity authority. Persisted profile identity fields become
identified compatibility evidence and migrate to normalized joins where
possible.

Clipped-shell builders are part of this scope. Their sidecars are legitimate
mapping/provenance evidence, but each sidecar and any donor Zarr must resolve to
the same recording before identity attrs are copied. In particular,
`utils/create_clipped_training_zarr.py` currently accepts a metadata donor
without a canonical same-recording comparison. Keep the evidence; remove its
ability to manufacture a competing identity.

### 4.4 First evidence and acceptance gates

Before mutations, run a dry-run census of manifest, root, `recordings`,
`datasets`, and profile identities. Report missing values separately from
conflicting non-null values.

Acceptance requires synthetic and fixture coverage for:

- manifest and Zarr agreement;
- manifest correction after an unchanged Zarr import snapshot;
- null inputs that do not erase known identity;
- conflicting non-null values that fail closed;
- idempotent repeated registration and maintenance;
- source-recording `datasets` and `recordings` projection parity;
- clipped sidecars that disagree with one another or with a donor Zarr and fail
  before output creation; and
- an explicit correction revision that succeeds once and rejects a stale
  compare-and-swap attempt.

## 5. Second implementation target: typed frame domains

### 5.1 There is no universal frame-count scalar

The existing `FrameDomain` vocabulary already distinguishes acquisition,
source-video, stored-Zarr, run, and crop-video axes
(`shared/frame_domains.py:22-30`). The defect is that legacy readers still ask
for an unqualified integer and the resolver records some conflicts as
diagnostic `missing` strings rather than raising (`:250-262,373-374`). It also
accepts zero in `_first_positive_int()` (`:173-184`) and currently installs the
same inferred count for both acquisition and source-video domains (`:492-503`).

The canonical count domains should be:

| Domain | Count authority |
|---|---|
| Source video | Independently validated source metadata or media probe bound to the source file. |
| Acquisition | Selected acquisition frame-clock/camera authority. |
| Stored Zarr | Physical stored array extent plus the declared stored-to-source mapping. |
| Run | The resolved run's explicit row axis/count and mapping. |
| Crop video | The explicit crop-video count/mapping. Supplemental rows remain acquisition-mappable when declared, but are intentionally unmappable to `crop_video_frame`; they are not part of crop-video count authority. |

Sampled/training archives use stored or run domains; they are not evidence that
`raw_video.total_frames` has one universal meaning. `duration_seconds` remains a
timing measurement and must not become a second frame-count authority through
implicit multiplication.

### 5.2 Proposed accessor shape

Readers should request a domain explicitly and receive a `FrameCountEvidence`
record rather than a bare fallback result:

```text
count
domain
source_kind
source_locator_or_path
source_profile
mapping_paths[]
conflicts[]
legacy_compatibility_used
```

Within one domain the resolver may have a declared source hierarchy, but it
must compare all independently available observations. Cross-domain counts are
not candidates in the same precedence ladder. When one legacy attr currently
seeds both source-video and acquisition counts, record one aliased observation
with provenance; do not report it as independent corroboration between domains.

### 5.3 Fix the source-metadata seed first

`preflight_source_video_metadata_backfill._proposed_source_metadata()` copies
plain root fields, including `total_frames`, into `source_video_metadata` when
the nested field is absent
(`utils/preflight_source_video_metadata_backfill.py:176-202`). On a materialized
archive, `import_video_metadata` writes root and raw `total_frames` from the
stored array length while keeping `source_video_total_frames` from source
metadata (`shared/import_video_metadata.py:450-527`). The authority parser then
validates the canonical form of that same nested blob, not an independent frame
observation (`shared/pixel_frame_authority.py:2167-2179`).

Therefore source metadata backfill must refuse to seed a source-domain count
from an unqualified attr unless profile evidence proves that attr is
source-domain. Prefer a bound media probe, canonical source metadata, or the
acquisition frame clock. Store any source-versus-stored difference as expected
domain evidence, not as a mirror mismatch.

### 5.4 Migration and acceptance gates

Migrate the opposite-precedence stimulus epoch resolver first, then acquisition
frame-clock publication, occupancy/crop planning, tuning, visualization, and
legacy utilities. Deprecate unqualified `total_frames`/`n_frames` consumption
only after shadow comparisons show each caller's intended domain.

Once the expected domain is explicit, enable the frame-clock writer's existing
row-count check. `publish_acquisition_frame_clock()` currently calls its source
validator with `expected_frame_count=None`
(`shared/acquisition_frame_clock.py:856-859`), bypassing the check implemented at
`:565-581`.

The minimum forcing fixture is:

```text
source/acquisition count = 5
stored count             = 3
stored -> acquisition    = [0, 2, 4]
legacy attrs             = intentionally contradictory
```

Tests must prove that source, stored, and run readers receive different correct
answers; no resolver silently selects the contradictory legacy attr; zero is
handled deliberately; and a missing mapping fails rather than assuming an
identity transform.

## 6. Ten-step execution plan

The following is the broader campaign order. Step numbers describe dependency
order, not a requirement that all work be one pull request.

### Step 1 — Freeze the taxonomy and run a read-only disposition census

Inventory every high-risk duplicate with semantic fact, current owner,
writers, readers, lifecycle phase, digest coverage, and intended disposition:
authority, evidence, projection, exact mirror, cache, receipt, supported
profile, migrate, or tombstone. Include a dry-run conflict census for identity,
frame domains, run selectors, status, geometry/calibration, epochs, and mirrors.

**Gate:** no proposed deletion or new authority proceeds without an explicit
disposition and a named consumer migration.

### Step 2 — Consolidate recording identity, including `datasets`

Land the read-only evidence resolver, conflict report, explicit correction
revision, and one registry projection writer. Route both normal registration
and maintenance through it, then normalize profile/status joins.

**Gate:** ordinary backfill cannot change a non-null identity or erase one with
null; explicit corrections are versioned, audited, and compare-and-swap
guarded.

### Step 3 — Consolidate typed frame domains and guard authority seeding

Make callers request source, acquisition, stored, run, or crop-video counts.
Fix `source_video_metadata` seeding before migrating readers. Raise on
same-domain conflict and on unavailable conversion edges.

**Gate:** the 5-source/3-stored fixture and shadow corpus report pass; no high-
risk migrated caller reads an unqualified count.

### Step 4 — Unify Zarr metadata view, run resolution, completion, and status

Route production publication and consumer opening through one lifecycle-aware
interface. Mutable/incomplete artifacts use direct metadata; published
immutable artifacts use and validate the consolidated generation. Low-level
writers and controlled diagnostics may open direct metadata explicitly, but
must declare that lifecycle mode rather than inherit an implicit default.
The generic `Recording.open()` path is in scope because it currently forces a
direct view through `shared/recording.py:167-174`; published consumers must not
inherit that choice accidentally.

Replace ambient `latest` logic with one typed run resolver whose named modes
preserve legitimate distinctions and return selection provenance.
Authoritative/latest scientific-consumption modes require both completion and
selector eligibility. Inventory modes may expose incomplete or ineligible runs
but cannot feed scientific consumption.

Include parentless `is_run_complete(run)` calls in the migration census.
Without the parent selector/generation context, those calls inherit legacy
acceptance and cannot establish authoritative consumption. Canonical paths use
the parent-scoped resolver with explicit strictness.

Collapse recording-step status writes into the shared ledger, including a
decision on whether `stale` is a status or structured detail. Eliminate raw SQL
writers that bypass normalization. A reverse-lexical fallback may remain only
as an explicitly named inventory mode, never as authoritative selection.

After identity and status rules are shared, collapse registry scientific
projection refresh into one explicit reconcile operation over immutable Zarr
evidence. `register_from_root()`, per-surface `refresh_*_from_root()` methods,
maintenance backfills, and `registry/inline_refresh.py` currently provide
separate orchestration/update policies even when they project the same
scientific publication.

**Gate:** direct and consolidated views agree for published selectors; every
run selection declares its mode; all recording-step status producers use the
allowed vocabulary and the same writer.

### Step 5 — Add one serial detection finalizer

Parallel detection publication owns Zarr completion, eligibility, and selector
activation. The serial closing operation consumes that immutable completion
evidence, reopens the exact selected run through the correct metadata view,
invokes the canonical detection validator and eligibility checks, and then
transactionally writes only registry current/history projections. It does not
repair or select Zarr as a side effect.

This is an ordered, lease-guarded, idempotent state machine, not an atomic
transaction across Zarr and SQLite. Pending/error evidence must survive a
failure between stores.

The existing `analysis_workflows/registry_finalize.py` closes serialized
derived-analysis stages. Raw detection is not currently covered by that
catalog/branch. Extend the canonical finalizer contract to raw detection rather
than creating a second detection-only approximation of it.

`emit_stage_completion()` currently catches all exceptions and returns `False`
(`registry/stage_complete.py:488-495`), while many callers ignore the result.
Failures must become durable pending/error evidence. A registry-write failure
does not necessarily invalidate scientifically valid Zarr bytes, but it must
prevent the registry projection from appearing successfully finalized.

**Gate:** no registry row can claim a detection run that the canonical Zarr
resolver rejects; retry is idempotent; partial failure remains visible.

### Step 6 — Unify arena geometry, calibration, and crop authority consumption

Route traditional detection, subject segmentation, arena assignment, and
registry calibration status through the selected geometry/calibration
authorities instead of legacy `dish_mask` or independent scalar ladders. Land
the crop resolver work from the crop audit: multiple supported crop profiles,
one position-authority interface, full validation per branch, and no
process-global monkey-patch adapter.

The selected arena contract deliberately does not write the legacy `dish_mask`
projection. Production readers cannot be switched by merely changing an attr
name: first give each reader a profile-aware selected-geometry path (or an
explicit checked compatibility projection), then remove the direct legacy
read. Likewise, registry calibration must call the full
`load_selected_calibration_snapshot()` path; its current scalar ladder can
report a present group as `ok` even when `usable_camera_scale` is false.

**Gate:** each supported profile has a real-writer-to-unpatched-reader CI test;
the registry cannot mark calibration usable when the canonical selected-
calibration loader rejects reciprocity or manifest binding, and status `ok`
alone is never treated as proof of a usable scale.

### Step 7 — Bind epoch projections, label speed semantics, and enforce mirror policy

Bind every copied epoch-window set to the canonical stimulus-epoch
manifest/selection digest in both the writer and loader. Required-epoch
workflows fail on resolution or binding errors; intentionally epoch-free
workflows persist an explicit not-applicable state rather than manufacturing
successful zero-window output.

Centralize speed measurement labels and the implementation of versioned
threshold/noise-floor policies per measurement product; do not impose one
universal threshold across unlike speed products. A raw centroid product may
remain, but its schema must bind `speed_source`, method, position source,
scale/calibration source, and units. Inventory root/raw-video
pairs: exact-compare true mirrors, remove unnecessary copies, and domain-stamp
facts that are not semantically equal. The native-detection metadata-file path
must read and compare both publication-status copies (or use an equivalent
metadata-file comparator); it cannot simply continue reading only `raw_video`.

**Gate:** stale epoch copies fail closed; every exported speed column declares
its source/measurement; no authoritative mirror reader bypasses its comparator.

### Step 8 — Version scientific metadata digest coverage by family

Define per-family versioned scopes for immutable scientific payload/metadata,
lifecycle and selector state, validation receipts, and operational telemetry.
Detection attrs consumed scientifically must move into a new sealed scientific
metadata contract. Lifecycle fields remain separately guarded; telemetry does
not enter the scientific digest. Preserve old manifests under their original
grammar rather than reinterpreting their digest.

New schemas should also converge on shared canonical JSON and array-digest
helpers with explicit algorithm identifiers. Persisted legacy digests keep
their original verifier; algorithm unification must never silently change the
preimage used to verify an old artifact.

**Gate:** mutation tests prove that scientific metadata drift fails, legitimate
lifecycle transitions remain possible, and old schema versions still validate
exactly as originally defined. Maintain an explicit per-family version matrix
for every accepted grammar (currently canonical detection v1-v3, refined
detection v1-v2, and subject-mask core v2-v5), including its original preimage
and verifier.

### Step 9 — Optimize workloads with receipt-backed bounded projections

Implement the Track audit's named bounded projection readers over the existing
verified authority. Read and hash each selected array once; keep the exhaustive
loader as the whole-publication audit path. Apply the same approach to other
families only when a receipt proves the same claim and has a real consumer.

Before treating a receipt as normal reader authority, choose and document the
receipt generation boundary: future publications only, an archive-bound
successor receipt for existing publications, or continued exhaustive reads.
Current validation receipts must not silently acquire a stronger reader claim
than their versioned contract grants.

Redesign Cluster 2 as separate records: an immutable, self-digested
publication-evidence record bound to artifact identity, manifest digest,
validator, owner, and phase; and mutable `cluster_output_staging` operational
telemetry. Do not bind the volatile staging document wholesale into scientific
manifests. A self-digest proves record integrity, not authenticity against a
writer with store access; replacement resistance still comes from the artifact
identity, owner/generation, immutable/authorized storage boundary, lifecycle
phase, and manifest binding.

Do not reuse `atomic_run_publisher.physical_copy.content_sha256` as the stable
scientific receipt. Its tree hash includes mutable metadata (`zarr.json`) and
is computed before later staging/lifecycle writes
(`shared/atomic_run_publisher.py:216-240,912-920,975-1041`). Reuse the payload-
receipt separation between decoded/physical payload and immutable metadata.

**Gate:** tamper/replacement tests fail closed. Each reader explicitly chooses
whether absent/stale evidence is a hard failure or invokes the exhaustive
loader; it never silently returns a partial projection. Every migrated Cluster
2 path has a real writer/evidence-writer-to-unpatched-reader test, and the final
staging snapshot remains until its current phase-specific readers are migrated
or explicitly retained. Benchmarks record selected metadata mode,
arrays/chunks touched, bytes, CPU, I/O, and wall time.

### Step 10 — Migrate compatibility surfaces, then delete superseded code

Perform parse-once manifest cleanup only inside proof-bearing loader boundaries;
retain source/parent/target and write-then-reopen validations. Remove adapters,
fallback writers, duplicate CLIs/helpers, and speculative APIs only after
call-site migration, dynamic-entry-point review, documentation updates, and
boundary tests.

`build_clipped_storage_keypoint_chain_fragments` remains tagged to the
keypoint-rebase decision rather than being deleted preemptively. Likewise,
supported crop profiles remain readable behind the resolver; coexistence is not
redundancy when semantics and workflows differ.

The other surveyed Cluster 5 entries also require explicit dispositions:

- `ALL_ENDED` has recovery semantics and test coverage; decide whether that
  recovery mode remains supported before removal.
- `EntityScope.CHASER` has a live planner branch and public export even though
  the current catalog does not instantiate it.
- `analyze_goodcopbadcop_immobility_artifact.load()` is called by its own CLI
  and tests. The entry point is broken, not dead; fix it or retire the complete
  diagnostic with its docs/tests.
- the two `build_ssh_bsub_runner` implementations have different default
  working directories. Deduplication must preserve that behavior explicitly.

**Gate:** no production or supported caller remains on the superseded boundary;
supported compatibility callers are routed through the shared resolver and
covered by CI. Tests pass without monkey-patching the removed boundary, and the
subtraction queue records the final disposition.

## 7. Code-reduction outcome and subtraction ledger

This plan is intended to make the production codebase smaller, not to place a
new authority layer beside every old implementation. Some steps need a short
add-then-migrate interval, but a step is not complete while its superseded live
writer, resolver, fallback ladder, adapter, or rescan remains without a dated
compatibility disposition.

| Step | Small canonical addition | Subtraction it must unlock |
|---:|---|---|
| 1 | Taxonomy and generated census only; no new runtime authority. | Delete proposals that have no remaining caller can proceed safely; duplicate inventories stop being maintained by hand. |
| 2 | Identity evidence type, correction record, and one projection writer. | Remove the maintenance `recordings` SQL implementation, competing source-precedence code, routine `datasets` identity mutation, and duplicated profile identity extraction where the shared resolver covers it. |
| 3 | Typed frame-count/domain result and guarded source seed. | Retire `stimulus_epoch_runs._resolve_dimensions`, `shared.metadata.get_total_frames`, local six-attr ladders, implicit identity conversions, and caller-specific count arithmetic as each caller migrates. |
| 4 | One lifecycle-aware opener, typed run/status resolver, and projection reconcile entry point. | Remove registry opener wrappers and implicit fallbacks, reverse-lexical authoritative selectors, the maintenance status SQL writer, and duplicate projection-refresh orchestration. |
| 5 | One serial detection closing path. | Remove detection-specific completion reconstruction, duplicated finalization fragments, and ignored best-effort registry completion calls once all producers use the closer. |
| 6 | Resolver branches for supported geometry/calibration/crop profiles. | Remove the process-global crop monkey-patch, legacy-reader special cases outside the resolver, independent calibration ladders, and production workflows' direct legacy `dish_mask` reads. |
| 7 | Bound epoch projection, named speed products, and one comparator per true mirror. | Remove swallowed-empty epoch fallbacks, repeated epoch-window parsers, duplicated speed kernels/policies where semantics match, and opposite root/raw fallback ladders. |
| 8 | Shared versioned digest helpers and explicit evidence scopes. | Remove per-family copies of canonical JSON/array hashing for new schemas; quarantine old algorithms in versioned compatibility verifiers instead of duplicating them in active writers. |
| 9 | Stable publication evidence and named bounded readers. | Delete only the deep rescans and broad array-loading paths proven equivalent by receipts; keep one exhaustive audit implementation. |
| 10 | No new authority. | Delete the remaining superseded CLIs, adapters, helper twins, speculative branches, and tagged dead APIs after their gates pass. |

Raw line count is not the only metric: a shared typed resolver plus stronger
tests can be longer than one unsafe fallback. The campaign should nevertheless
be **net-negative in production code**. Tests, fixtures, and migration evidence
may grow. Each implementation handoff should report:

- production lines added and removed;
- live implementations before and after;
- migrated call sites and deleted fallback sites;
- compatibility branches remaining, with owner and retirement condition;
- full-tree scans or broad array loads before and after; and
- tests/docs added to preserve the stronger boundary.

An add-only pull request can be an intermediate migration commit, but the
campaign item remains open until its paired deletion lands. If a former path
must remain for old artifacts, concentrate it behind the shared resolver and
count it as a bounded compatibility branch, not a second production authority.

## 8. Cross-cutting safety and acceptance gates

The campaign is complete only when each affected semantic fact satisfies all
applicable gates below.

1. **Read-only evidence first.** Corpus census and shadow comparison precede
   mutations, especially for identity, frame counts, selectors, and registry
   reconciliation.
2. **Conflict is a first-class result.** Missing, conflicting, legacy-derived,
   and verified states are not collapsed into one nullable value.
3. **Trust boundaries remain explicit.** Local candidate, hidden copy, renamed
   path, completed run, consolidated publication, selector activation, and
   registry projection are distinct phases.
4. **Published metadata is lifecycle-correct.** Writers and mutable readers use
   direct metadata. Consolidation is the final visibility step for immutable
   publication, and published readers validate the consolidated generation.
5. **Compatibility is tested, not assumed.** Each supported grammar has a
   real-writer-to-unpatched-reader test. An adapter is not evidence that a
   profile is fully supported.
6. **Receipts have bounded claims.** Integrity, scientific validation,
   publication, selector eligibility, and telemetry remain distinguishable, as
   required by
   [`zarr_payload_validation_receipt_contract.md`](../zarr_payload_validation_receipt_contract.md).
7. **Registry repair is staged.** Scheduled read-only comparison and durable
   reports come before any scheduled actuator. Repairs use the canonical writer
   and the Palette SQLite runtime.
8. **Required CI is blocking.** Failed, cancelled, timed-out, or skipped
   required checks leave a branch explicitly incomplete and not merge-ready.
9. **Performance cannot weaken authority.** A bounded projection must name its
   consumption shape and verify the selected authority. There is no generic
   `skip_validation` mode.
10. **Deletion is last.** Static grep alone is insufficient for public APIs,
    command entry points, recovery modes, and configuration-driven branches.

## 9. Deferrals and non-goals

- This document does not select a winning raw precedence for recording identity
  or frame count. Those facts need typed authority/evidence models.
- It does not create one universal `latest`, one universal frame scalar, or one
  universal speed value.
- It does not make the registry authoritative for scientific arrays or derived
  measurements.
- It does not bind mutable `cluster_output_staging` telemetry into immutable
  scientific manifests.
- It does not remove Track's decoded, physical, or immutable-metadata roots.
  They prove different claims; workload optimization changes how bounded
  consumers use verified evidence.
- It does not delete supported crop publication profiles or use a resolver as a
  weaker adapter. Each branch retains full validation.
- It does not authorize scheduled automatic repair, selector activation,
  production publication, or compatibility deletion. Those require the
  implementation and acceptance evidence above.

## 10. Source map

The main governing contracts and implementations for this plan are:

- Registry authority and immutable identity policy:
  [`registry_data_governance_policy.md`](../registry_data_governance_policy.md),
  `registry/db.py:606-612,730-761,2534-2536,2601-2618`, and
  `registry/maintenance.py:824-948`, with joined-view identity sources at
  `registry/migration_bodies.py:4430-4477,6580-6582,6760-6764`.
- Identity normalization history:
  [`recording_registry_normalization_todo.md`](../recording_registry_normalization_todo.md)
  and
  [`acquisition_batch_registry_contract.md`](../acquisition_batch_registry_contract.md).
- Clipped-shell identity evidence and donor binding:
  `utils/create_clipped_analysis_zarr.py:536-560,584-591,661-679` and
  `utils/create_clipped_training_zarr.py:52-69,285-327,474-532,620-622`.
- Frame-domain design and current implementation:
  [`frame_domains_resolver_design.md`](../frame_domains_resolver_design.md),
  [`source_video_metadata_contract.md`](../source_video_metadata_contract.md),
  `shared/frame_domains.py:22-30,173-184,250-262,373-374,492-503`, and
  `analysis/stimulus_epoch_runs.py:132-145`; authority seeding and verification
  at `utils/preflight_source_video_metadata_backfill.py:168-202`,
  `shared/import_video_metadata.py:393-535`, and
  `shared/pixel_frame_authority.py:2858-3075`.
- Run-resolution semantics:
  [`run_resolution_semantics.md`](../run_resolution_semantics.md),
  `shared/run_resolution.py:20-44,73-79,175-305`,
  `shared/zarr_run_completion.py:427-443,506-681`, and
  `registry/maintenance.py:4097-4134,4164-4194`.
- Zarr metadata lifecycle:
  `shared/zarr_io.py:14-48`, `registry/db.py:206-216`,
  `registry/maintenance.py:132-142`, `registry/chaser_metadata.py:85-90`, and
  `registry/stimulus_metadata_backfill.py:80-85`, plus the compatibility opener
  in `shared/zarr_helpers.py:413-425` and its generic caller at
  `shared/recording.py:167-174`.
- Recording-step status and completion:
  `registry/status_ledger.py:11-12,105-244`,
  `registry/stage_complete.py:488-495`,
  `registry/maintenance.py:5311-5412,6917-7013`, and
  `registry/migration_bodies.py:3242-3261`.
- Existing serialized-stage finalizer and detection coverage gap:
  `analysis_workflows/registry_finalize.py:1-121,310-633` and
  `analysis_workflows/storage_contract_catalog.py:227-451`.
- Registry scientific-projection refresh paths:
  `registry/db.py:5320,5983,6619-6787`,
  `registry/maintenance.py:2560-4055`, and
  `registry/inline_refresh.py:1-166`.
- Epoch and speed divergence:
  `analysis/chaser_distance_runs.py:528-545`,
  `analysis_workflows/resolved_epoch_selection.py:450-620`, and
  `utils/export_cross_recording_analytics.py:3262-3324`.
- Exact-mirror example and bypass:
  `shared/acquisition_publication_status.py:280-308` and
  `cluster/native_detection_authority.py:76-95`.
- Geometry, calibration, and crop authority:
  `analysis_workflows/materializers/arena_geometry_candidates.py:1-5,222`,
  `analysis_workflows/materializers/arena_geometry_selection.py:342-480`,
  `detection/detect_traditional.py:280-302`,
  `segmentation/subject_segmentation.py:226-278`,
  `tracking/arena_assignment.py:362-446`,
  `shared/selected_calibration.py:2946-3003,4044-4325`, and
  `registry/maintenance.py:3969-4016`.
- Versioned scientific metadata digest scopes:
  `shared/zarr/canonical_detection_manifest.py:568-597`,
  `shared/zarr/refined_detection_manifest.py:320-350`,
  `shared/zarr/subject_mask_core_publication.py:690-724,829-895`, and
  `analysis/chaser_component_publication.py:390-440,559-609`.
- Receipt scopes and Track publication boundaries:
  [`zarr_payload_validation_receipt_contract.md`](../zarr_payload_validation_receipt_contract.md),
  `shared/atomic_run_publisher.py:216-240,912-920,975-1074`,
  `shared/zarr_payload_receipt.py:519-742`,
  `analysis/track_kinematics.py:2953-2960,11903-12043,12450-12473`, and
  `analysis_workflows/materializers/track_kinematics.py:638-746`.
- Acquisition clock publication check:
  `shared/acquisition_frame_clock.py:565-581,856-859`.
- Bounded projection optimization and acceptance criteria:
  [`track_motion_reader_optimization_2026-08-24.md`](track_motion_reader_optimization_2026-08-24.md)
  §§4-9.
- Compatibility migration and deletion gates:
  [`pipeline_survey_2026-08-24.md`](pipeline_survey_2026-08-24.md) §4,
  [`crop_contract_split_audit_2026-08-24.md`](crop_contract_split_audit_2026-08-24.md)
  §§8-9, `cluster/clipped_storage_finalization.py:260-298`, and
  `shared/registry_stage_complete.py:1-20`.
