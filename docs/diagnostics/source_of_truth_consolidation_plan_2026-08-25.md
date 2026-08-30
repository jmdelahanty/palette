# Source-of-Truth Consolidation Review and 10-Step Plan

**Date:** 2026-08-25

**Plan disposition:** governing architecture and recording-identity
implementation record. Cross-cutting authority/admission execution status is
owned by
[`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md).
The ordered current-v2 recording-identity packages in §4.7 remain the scoped
implementation checklist for that workstream; downstream producer admission is
not part of migration 73. Stage-level execution and supervisor-review
dispositions are maintained in the active queue rather than duplicated here.

**Method:** eight parallel read-only Luna xhigh reviews of recording identity,
frame authority, epochs and speed, run selection and completion, mirrors and
manifest coverage, registry reconciliation, omitted authority paths, and the
cross-cutting architecture, followed by direct source confirmation, the
Step 1 recording-identity census, and parallel review of that evidence. Six
additional parallel read-only Luna xhigh inventories reviewed the committed
Step 2 checkpoint for writer access, subtraction, safety, activation, plan
drift, and next-stage sequencing.

**Original review state:** branch
`agent/palette/clipped-geometry-acquisition-authority-20260821`, HEAD
`e58443c3` (`docs: record track reader optimization audit`).

**Implementation checkpoint:** branch
`agent/palette/recording-identity-evidence-20260825`, HEAD
`6969043ef801b2a01028f28f9da9194b31aa9924`
(`registry: add recording identity safety checkpoint`). Step 1 remains
committed at `d816771d11cb`. Step 2 now has a committed
`palette.source_recording_identity.v2` producer profile
and one complete `palette.source_recording_identity_claim.v2` over current Zarr
v3 source roots, an initial authority projection, migration-73 ledger,
immutable `palette.recording_import_receipt.v2` binding, verified
identity/import readers, paired-profile fail-closed classification, and
prune/dedupe fences. The semantic claim is locator-free: target paths and
acquisition references live in the receipt and guarded registry binding. New
organizer/importer paths agree on one per-camera `recording_id`, one shared
acquisition `session_uuid`, explicit `camera_id`, and exact root
classification; newly created profiled artifacts use this contract while
unmarked historical artifacts remain on the deferred compatibility path.
Central current-profile generic registration is fenced; the organized importer and
single-recording pipeline call the `shadow_synchronize_recording_import()`
gateway through a designated-writer-host boundary, bound reconciliation uses
the verified reader, and maintenance rejects an unbound current profile before
its legacy identity SQL. Current publication consolidates metadata only after
the final identity/classification checks and validates the consolidated
generation before minting the receipt. Three historical subject-repair tools
now reject current-v2 before mutation, and the detection-training reader
requires receipt-bound registry verification.

The importer lifecycle is intentionally one-shot: clean committed producer
identity is checked before archive creation, existing receipt evidence seals the
source-import surface against re-import, fresh registry finalization occurs
before inference, and bound pipeline replay invokes no import writer. Derived
versioned runs may still be appended; the receipt does not claim whole-archive
immutability. A focused integration crosses real importer orchestration and
actual Zarr/acquisition publication through receipt minting, registry
projection, close/reopen, and the newly added verified reader; media probing,
frame-clock application, and Git identity are controlled test doubles.

At `6969043ef801b2a01028f28f9da9194b31aa9924`, 328 focused tests passed
outside the sandbox on `2026-08-25`; the final 112-test rerun also passed. The
unrelated Track audit edit and Sleepyfish handoff are outside this checkpoint.
Required repository CI has not run, so this branch is incomplete and not
merge-ready. Issuer authorization, operational enforcement that all canonical
registry writers use the designated-host shadow gateway (including quiescence
or fencing of noncooperating writers), durable rejected-attempt and publication
evidence, implementation compaction and deletion work, a quarantined physical
canary, and required CI remain open activation gates. Identity correction,
locator movement, and current clipped-output support are deliberately
unsupported for the initial immutable v2 profile and remain separate future
contracts, not silent legacy fallbacks.

**Companion audits:**
[`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md),
[`redundancy_campaign_2026-08-24.md`](redundancy_campaign_2026-08-24.md),
[`pipeline_survey_2026-08-24.md`](pipeline_survey_2026-08-24.md),
[`track_motion_reader_optimization_2026-08-24.md`](track_motion_reader_optimization_2026-08-24.md),
[`crop_contract_split_audit_2026-08-24.md`](crop_contract_split_audit_2026-08-24.md),
[`clipped_eye_assignment_authority_failure_2026-08-25.md`](clipped_eye_assignment_authority_failure_2026-08-25.md),
[`contract_enforcement_divergence_review_2026-08-21.md`](contract_enforcement_divergence_review_2026-08-21.md),
[`subtraction_queue_2026-08-21.md`](subtraction_queue_2026-08-21.md), and the
implemented Step 1 evidence record
[`recording_identity_census_2026-08-25.md`](recording_identity_census_2026-08-25.md).

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

The recording-identity census is now a bounded evidence baseline, not a
writer-generation census. Its `explicit_source_layout` predicate selects
active source-recording analysis artifacts with registry layout markers; it
does not identify the Palette commit that produced them. The identity slice of
Step 1 is complete, but the remaining fact-family census and proof of the
current writer remain open. Step 2 therefore starts with synthetic
writer-to-unpatched-reader tests and a commit-pinned source canary held outside
production locators and authority, not with repair of the observed corpus.

The intended shape is:

```text
observed evidence from manifests, Zarr, probes, and ledgers
                              |
                              v
           one typed, fail-closed resolver per semantic fact
                              |
              +---------------+----------------+
              |               |                |
        resolved fact   bound projection   exact mirror
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

This taxonomy produces seven immediate rules:

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
6. A schema or layout marker identifies a contract shape, not the exact code
   revision that produced an artifact. Producer generation requires explicit,
   immutable producer evidence.
7. A bounded diagnostic that reaches an observation, traversal, or cardinality
   cap is incomplete and must fail closed. Such a cap is not a scientific data
   threshold and must never be interpreted as a frame-row limit.

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
| The 31 artifacts selected by the recording census represent the current implementation. | They are the `explicit_source_layout` evidence cohort: 23 `analysis_zarr` rows and 8 `rolling_clips` rows selected by registry layout markers. The predicate does not establish writer generation. Batman's 36 and Goodbatbadbat's 84 active source-analysis rows are excluded because those registry markers are null; their `recording_analysis_v1` root schema marker does not prove that they are legacy or came from a different writer. |
| Eight million-row frame indexes imply 31 long recordings or duplicated frame data. | The 31 count is per-camera source-artifact/registry entities, not acquisition sessions or frame indexes. The 8 Parquets are two acquisition families times four cameras. Their 1,188,000- or 2,937,604-row counts are expected; row count alone is not a defect. A separate whole-file hash check found no duplicate Parquet files. |
| Existing downstream provenance proves which Palette importer created a recording root. | Modern `run_provenance` and `stage_provenance` often bind downstream code, config, inputs, and runtime, but they identify the downstream producer. Root schema/layout markers and the acquisition producer label do not bind the exact Palette importer commit. New source publication needs its own immutable import receipt. |
| A run marked `complete` necessarily passed producer-provenance validation. | Parentless completion skips the strict parent-scoped provenance gate. A bounded Goodbat review found 355 selector-ineligible, unselected runs marked complete with `run_provenance.git_sha=null` and one failed run. None was authoritative, but the completion contract still has a loophole. |
| The ordinary `recordings` upsert freezes an existing `session_uuid`. | `COALESCE(excluded.session_uuid, recordings.session_uuid)` at `registry/db.py:2638-2640` lets any non-null incoming value overwrite the existing value; it preserves the existing value only when the incoming value is null. The maintenance writer at `registry/maintenance.py:961-963` hard-overwrites and can also erase with null. Both are last-writer-sensitive. |
| The identity problem is confined to `recordings`. | `datasets` is also mutable in conflict with policy: `session_uuid=excluded.session_uuid` and a non-null incoming `recording_id` wins at `registry/db.py:2554-2557`. Governance declares `datasets.session_uuid` immutable generally and `datasets.recording_id` immutable for `artifact_kind='source_recording'`. Joined views can therefore consume identity from different registry copies. |
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

### 4.1 Implemented evidence baseline and its limits

The read-only census committed at `d816771d11cb` establishes the first bounded
baseline. Its default `explicit_source_layout` cohort selected 31 active
source-recording analysis artifacts with registry layout evidence:

| Evidence slice | Result | Interpretation |
|---|---:|---|
| `analysis_zarr` rows | 23 | Per-camera source-artifact/registry entities, not acquisition sessions. |
| `rolling_clips` rows | 8 | Two acquisition families times four cameras. |
| Frame-index Parquets | 8 | All identity projections were single-valued and complete. Million-row indexes are expected. |
| Zarr artifact scopes | 28 complete, 3 incomplete | The three incomplete scopes contain non-finite JSON `Infinity`; they were not accepted as evidence. |
| Findings | 24 action-required, 23 expected | Action-required: 8 artifact conflicts, 5 missing artifact identity fields, 8 recording-sidecar conflicts, and 3 incomplete Zarr scopes due to non-finite metadata. The 23 expected findings record `dataset_id == session_uuid`. |

The eight clipped artifacts describe two four-camera failure families, not 24
independent root causes. The census chooses no precedence winner, emits no
effective identity, authorizes no repair, and reports
`proves_writer_generation=false`. Its v5 evidence binds registry snapshot
SHA-256
`f37c8f2155904becd2a61a613b1e6036fa5d58b02c505cc536f5136d2c598d1b`
to report-body SHA-256
`5f35c874f3cab45cfa0f7481e5b6a6f9bafcb9ddba3a4b8adf5d791ba5444420`;
the detailed interpretation and command are in the companion census document.
Focused census validation at that commit passed 20 tests, `py_compile`, and
`git diff --check`. Required repository CI remains unrun.

The default 100,000 observation cap bounds accumulated identity observations
within a metadata scan scope. It is not a frame-row, Parquet-row, array-size,
or artifact-count limit. Zarr traversal and Parquet distinct identity
cardinality have separate bounds. Reaching any bound makes the affected scope
incomplete and the command non-successful; observation/traversal coverage caps
are `unresolved`, while identity-cardinality overflow is `action_required`.
Absence of a conflict cannot be inferred from capped evidence. No selected
live scope reached those bounds. The three incomplete scopes failed because of
`Infinity`, not scale.

Batman's 36 and Goodbatbadbat's 84 active source-analysis rows are outside this
cohort because both registry layout markers are null. All 120 roots carry the
`recording_analysis_v1` schema marker, but that marker identifies a data
contract, not a writer commit. Their compatibility and repair disposition is
deferred; they must not be labeled legacy merely from this exclusion.

Downstream `palette.run_provenance.v1` and stage provenance often identify the
exact producer of a derived run. They do not identify the Palette importer that
created the recording root. `source_layout`, `source_frame_index_schema`,
`recording_analysis_v1`, and versioned acquisition producer labels likewise do
not bind an exact Palette code revision. Current-writer safety therefore cannot
be inferred from this corpus.

### 4.2 Why precedence cleanup is insufficient

`resolve_dataset_id()` prefers existing Zarr identity attrs and consults the
manifest only when they are absent (`registry/db.py:618-624`). Normal recording
context extraction reads root attrs, embedded context, and the manifest in that
order (`registry/db.py:742-818`). The maintenance backfill reads the manifest
without opening the Zarr (`registry/maintenance.py:785-948`).

The fallback also treats `recording_manifest.recording_id` as a candidate
`session_uuid` when the latter is absent (`registry/db.py:620-623`). The new
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
  a primary key. A four-camera acquisition has four distinct recording
  entities and therefore four distinct `recording_id` values.
- `session_uuid` is acquisition-session identity. It may be shared by the four
  camera recordings from one acquisition, may participate in matching, and is
  not a sufficient registry primary key. It must not be silently changed. Two
  four-camera sessions therefore produce eight `recording_id` values grouped
  under two `session_uuid` values.
- `organizer_recording_id`, `orange_session_id`, directory names, and legacy
  session/family labels are source context unless a versioned producer contract
  explicitly maps one of them to a canonical fact. They must never become
  `recording_id` or `session_uuid` through fallback precedence.
- Zarr root identity attrs should be treated as import-time artifact evidence,
  not a mutable correction ledger. Current paths tend to freeze them, but they
  are not cryptographically or structurally immutable merely because they are
  attrs.
- `recordings`, `datasets`, profile tables, and status views are registry
  projections over one resolved identity decision.
- The initial current-v2 contract does not correct an established identity. A
  conflicting manifest, root, registry row, or locator fails closed. If
  correction support is later enabled, it must use append-only revisions with
  actor, reason, evidence, prior revision, and compare-and-swap protection. The
  acquisition-batch assignment pattern in
  [`acquisition_batch_registry_contract.md`](../acquisition_batch_registry_contract.md)
  is the in-repo model.

### 4.3 The unified claim and fail-closed profile classifier

The current implementation has one complete `SourceRecordingIdentityClaim`,
not a second identity-resolution object. The shared loader reads the
two declared source roles—strict bounded manifest JSON and direct Zarr-root
metadata—and returns a claim only after their identity and current-source
classification agree:

```text
SourceRecordingIdentityClaim
  schema_id: palette.source_recording_identity_claim.v2
  canonicalization: canonical-json-sha256
  identity:
    recording_id
    session_uuid
    camera_id
    recording_id_mapping_profile?
  verified_source_roles:
    recording_manifest
    zarr_root_direct_metadata
  root_classification:
    artifact_schema_id = recording_analysis_v1
    artifact_kind = source_recording
    zarr_origin = source
    zarr_use = analysis
    zarr_purpose = analysis
  claim_sha256
```

The claim is deliberately locator-free. `recording_manifest` and
`zarr_root_direct_metadata` name semantic source roles, not filesystem paths;
the target-relative path and acquisition ownership/frame references belong to
the import receipt, while the canonical dataset locator and claim digest belong
to the guarded registry binding. The claim digest seals the normalized
identity/classification claim, not the full mutable source documents.

`load_source_recording_identity_profile(root)` classifies the paired manifest
and direct root together. A current declaration on only one side, a current
declaration with mismatched identity, or an exact current declaration without
the required source classification fails closed. A current manifest may sit
above an explicitly non-source sibling artifact; an entirely unprofiled pair
is the explicit compatibility state. This prevents a stripped or one-sided
current root from silently falling through to a legacy writer.

The projection writer and verified reader compare the stored claim with the
live paired claim. There is no approved-correction input in v2. An absent value
may be filled only when every required, same-semantic source is present and
agrees. Two conflicting non-null values produce a conflict, not a precedence
winner; unresolved results do not carry a resolved identity. Exact producer
and source-import fingerprints belong in the separate receipt.

### 4.4 One projection writer and a separate import receipt

Replace the maintenance raw SQL and normal registration divergence with one
writer that:

1. consumes only a successfully validated identity claim;
2. creates a missing projection or fills an unambiguous null;
3. rejects every conflicting non-null identity; a future correction API must
   remain separate from routine projection;
4. applies the same rules to `recordings` and `datasets`;
5. never changes a primary identity through ordinary rescan/backfill;
6. does not rewrite `dataset_id` or change path-hash identity as a side effect
   of resolving recording/session identity;
7. records the claim digest, authority revision, and projection timestamp; and
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

New source publications now use the immutable
`palette.recording_import_receipt.v2`, which binds:

- the current importer/profile identity, exact clean Palette commit, and
  canonical import-configuration hash;
- one embedded `palette.source_recording_identity_claim.v2`, rather than a
  second identity document;
- the existing acquisition-import ownership and camera-frame record references
  and digests, which bind source metadata and frame semantics; and
- the normalized target-relative path plus the receipt schema/producer
  version.

The semantic identity claim contains no locator or second publication
authority. Locator ownership is split explicitly: the receipt binds the import target relative to
the producer's declared root, and the guarded registry binding binds the
canonical dataset path, claim digest, and receipt digest. This keeps path
movement and identity correction out of the immutable initial v2 claim.

That receipt is producer evidence, not a second identity authority. A new
publisher that cannot establish its code identity must fail or create an
explicitly quarantined source artifact under a root-level publication-
eligibility contract that production readers and registry authority reject.
Historical missing receipts remain `unknown`; downstream run provenance must
not be used to backfill them.

Keep the receipt bounded. It must not copy the full manifest, growing root
`zarr.json`, media metadata, frame-index rows, or acquisition authority
records. Those payloads retain their existing owners; the receipt references
their immutable digests. Mint the receipt only after final import validation,
write it with exclusive-create/exact-replay semantics, and bind its digest in
the guarded registry projection. A self-digest establishes integrity, not
authentication against an actor who can write both the artifact and registry.

The direct mutator in `registry/repair_recording_identities.py` remains
deferred. If retained, it must become a client of the explicit correction
revision and projection writer; it cannot remain an independent repair
authority.

### 4.5 First evidence and acceptance gates

The committed census is the failing-before-change baseline. It reports missing
values separately from conflicting non-null values, chooses no winner, and
authorizes no corpus mutation. Existing-artifact repair, including the
unmarked Batman and Goodbatbadbat cohort, remains deferred while the current
implementation boundary is consolidated.

Initial current-v2 activation requires synthetic and fixture coverage for:

- manifest and Zarr agreement;
- null inputs that do not erase known identity;
- conflicting non-null values that fail closed;
- idempotent repeated registration and maintenance;
- source-recording `datasets` and `recordings` projection parity;
- a regular-source real-writer-to-unpatched-reader round trip that validates
  the import receipt and registry projection;
- both known four-camera conflict families failing before output creation when
  sidecar, donor, root, camera, or frame-map bindings disagree;
- missing root `session_uuid` and non-finite metadata remaining unresolved,
  never accepted through fallback; and
- a commit-pinned source canary, held outside production locators and authority
  under an explicit quarantine contract, proving the consolidated current
  writer/reader boundary before any production publication or repair is
  enabled.

Before correction, relocation, or clipped-output support is enabled, each
requires its own additional acceptance boundary: stale compare-and-swap and
multi-dataset correction tests; audited locator-transition and in-flight-reader
tests; or sidecar/donor/camera/frame-map equality tests, respectively. Deferring
those capabilities means rejecting them, not routing them through legacy
precedence.

### 4.6 Step 2 implementation checkpoint and live checklist

This checkpoint wires selected current-profile call paths, but the branch
remains non-production and ineligible for production activation. It proves the
initial identity boundary without claiming that a caller holding artifact and
registry write access is an authenticated publisher. Historical-artifact
compatibility and corpus repair remain deferred, but current activation gates
are not deferred.

| Work item | Status at this checkpoint | Remaining gate |
|---|---|---|
| Exact current-source profile and identity claim | Implemented for new profiled artifacts | `palette.source_recording_identity.v2` defines one per-camera `recording_id`, a shared acquisition `session_uuid`, explicit `camera_id`, and exact current-source root classification. The shared loader emits one locator-free `palette.source_recording_identity_claim.v2` with the normalized identity, verified source roles, root classification, and `claim_sha256`; its optional deterministic session/camera mapping claim is sealed and checked, while independently supplied IDs remain valid without that claim. Newly created organizer and analysis-import outputs use the shared strict contract; existing unprofiled roots retain an explicit compatibility path and are not inferred or backfilled. |
| Paired-profile classifier and typed resolver | Implemented at the current v2 boundary | Manifest/root profile declarations are classified together. One-sided current declarations, mismatched claims, and current roots without exact source classification fail closed; an entirely unprofiled pair is the explicit compatibility state. Required facts remain distinct and there is no precedence or cross-fact fallback. |
| Final consolidated current publication | Implemented final step; activation hardening open | Current source publication consolidates only after payload, identity, classification, and acquisition/crop checks complete; it reopens the consolidated view and rechecks the claim and classification before minting the receipt. Before activation, bind or recheck a stable generation across consolidation and receipt minting, and make every mutable importer open explicitly unconsolidated. |
| Migration-73 authority ledger | Implemented, opt-in | No authority rows are backfilled. The migration creates the ledger tables, indexes, triggers, and schema fingerprint; evidence/revision/receipt rows are append-only, current and dataset pointers are exact, the revision trigger is contiguous, and concurrent migration version is rechecked under the write lock. |
| Initial projection writer | Implemented, opt-in | Creates/fills only recording/session identity, never lifecycle classification or locator moves; exact replay is idempotent; it does not yet authorize correction. |
| Verified bound-identity reader | Implemented and routed for current-v2 | Rechecks dataset, recording, current pointer, latest revision, stored evidence, live manifest/root evidence, bound receipt, and acquisition authority. Bound scan/reconcile, maintenance, the pipeline replay, and detection-training input now use it. Direct canonical NFS publication for the two current import routes remains disabled by the activation gate; `registry_rescan --safe-shadow-publish` remains an activation-blocking bypass until it uses the designated-host gateway. |
| Generic prune and dedupe safety | Implemented | Authority-bound datasets are excluded from generic mutation; dedupe rechecks bindings under `BEGIN IMMEDIATE`; FK `RESTRICT` remains the final backstop. |
| Strict source and relational adversarial tests | Expanded focused suite green | The latest claim, authority, receipt, importer, organizer, pipeline, reconciliation, schema, submitter, profile, and historical-tool boundary runs passed 328 focused tests outside the sandbox on `2026-08-25` at `6969043ef801b2a01028f28f9da9194b31aa9924`; the final 112-test rerun also passed. They cover malformed input, transaction-fence mutation, exact replay, append-only state, parent/current corruption, receipt/acquisition tampering, close/reopen verification, sealed re-import, early finalization, bound replay, two-session/eight-camera identity grouping, designated-host submission, and pre-mutation compatibility fences. Required repository CI remains unrun. |
| Exact producer/import receipt | Implemented, routed, not issuer-authenticated | `palette.recording_import_receipt.v2` is a bounded, digest-named immutable sidecar bound append-only in migration 73. It embeds exactly one `palette.source_recording_identity_claim.v2`, plus the clean producer commit, invocation-configuration digest, target-relative path, and existing acquisition ownership/frame references and digests. It does not turn the semantic claim into a locator-bearing object. The projection verifies the receipt sidecar before entering its write transaction and calls live acquisition/receipt verification immediately before and after projection writes. Shadow publication adds host, lock, candidate, backup, and canonical-registry hash fences. The configuration digest is invocation provenance, not proof of stimulus/setup outputs. The self-digest is not issuer authorization against an actor with both artifact and registry write access. |
| Source-import receipt lifecycle | Initial one-shot policy implemented | Clean producer state is checked before writers run; any existing valid receipt sidecar seals the current source-import surface; bound pipeline replay skips import; fresh current identity is finalized before downstream inference. Receipt replacement, overwrite of a sealed current source, and new publication generations are rejected rather than improvised. Exact recovery of an unbound receipt remains a quarantine/manual-finalization task. |
| Real importer round trip | Current identity matrix green; physical canary still required | One focused test crosses real importer orchestration and actual Zarr/acquisition publication, receipt minting, registry projection, close/reopen, and `read_verified_recording_import()` with media probe, frame-clock, and Git test doubles. A second test uses the real draft-manifest, organizer, analysis-root, receipt, registry-sync, and verified-reader paths for two sessions times four cameras: it proves eight unique `recording_id` values, two shared `session_uuid` values, four cameras per session, and eight exact receipt bindings. External media and Git probes remain controlled doubles, so a commit-pinned physical canary is still required before production activation. The reader is new, so this is not an “unpatched-reader” compatibility proof. |
| Explicit correction path | Deferred and unsupported by current v2 | The ordinary writer accepts only the initial immutable decision and exact replay. Conflicts fail before mutation. Do not expose correction semantics until a separate stable-scope CAS contract, multi-dataset rebinding, durable reason/evidence, and correction-aware reader are implemented and tested. This is not an activation blocker for an explicitly immutable current-v2 publication. |
| Locator transition | Deferred and unsupported by current v2 | Migration 73 now rejects direct identity or locator mutation for bound datasets. Initial projection binds the canonical locator; relocation requires a future dedicated CAS/history contract. This is not an activation blocker while current-v2 publication explicitly forbids moves. |
| Durable conflict report | Not implemented | Current conflicts fail before mutation but are not persisted. A rejected-attempt report must commit outside the rolled-back projection transaction. |
| Multi-host registry writer boundary | Implemented for two current import routes; activation blocked | Organized import and the single-recording pipeline call `shadow_synchronize_recording_import()` through an explicit designated-writer-host check, host-local mutex, node-local candidate, durable backup, and existing NFS lock/hash publication fences. The gateway never opens the canonical registry writable. This is not yet a global funnel: noncooperating direct writers can race the final hash/replace interval and must be quiesced or fenced; issuer authorization remains open. |
| Normal registration, maintenance, and refresh routing | Partially routed; activation blocked | Current import routes use the gateway and the central `synchronize_recording_import()` dispatch; bound reconcile and maintenance use verified authority, while unbound maintenance fails before legacy SQL. Disabling the projection-refresh submitter's `--apply` mode covers only one path. `registry_rescan --safe-shadow-publish` still bypasses the designated-host gateway, and scan/rescan/reconcile, stage completion, inline refresh, maintenance/admin tools, and derived/training finalizers still contain direct writable registry entry points. Inventory and fence `scan_zarr()`, `refresh_bound_current_source_import()`, `reconcile_dataset_from_root()`, legacy `refresh_*_from_root()` methods, maintenance backfills, and `inline_refresh`; migrate read-only consumers to query-only connections. Remaining work is global writer quiescence/fencing, authorized issuance, durable rejected-attempt/publication evidence, and deletion of superseded current identity SQL after all supported current callers pass. |
| Clipped sidecar/donor binding | Not started | Keep deferred compatibility separate, but current clipped output must not activate until sidecar, donor, recording/session, camera, and frame-map bindings use the authority. |
| Historical corpus repair | Deferred by decision | No backfill or repair of Batman, Goodbatbadbat, the four-camera conflict families, or other old archives in this implementation slice. |
| Generated schema reference | Current | Regenerated for migration 73 (63 tables, 55 views, 3,169 columns); the standalone generator `--check` passes. Required repository CI is still unrun. |

The committed checkpoint is still substantially additive. Relative to
`bf8b7c6d188e`, tracked production and script files add 4,295 lines and remove
322, net **+3,973 lines**. The four new foundational modules
(`source_recording_identity.py`, `recording_import_receipt.py`,
`recording_identity_authority.py`, and `shadow_publish.py`) contain 2,596 lines.
The organized importer itself is net -69 lines, and several current call sites
are routed, but the campaign has not made the repository smaller yet. This is a
safety implementation, not yet a streamlining win. Step 2 must not be counted
as consolidation until the single-writer boundary displaces the direct writer,
the maintenance and normal-registration identity implementations are deleted,
and the authority/receipt code receives a substantial compactness pass.
Correction, relocation, and receipt-generation ledgers are intentionally not
being added to this slice because they would expand surface area without
serving the supported immutable current-v2 workflow.

The 328-case focused checkpoint used the Palette Python runtime across bounded
invocations. The combined file set was:

```text
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_recording_identity_authority.py \
  tests/unit/fisheye/test_recording_import_receipt.py \
  tests/unit/fisheye/test_recording_import_authority_integration.py \
  tests/unit/fisheye/test_import_recording_analysis.py \
  tests/unit/fisheye/test_source_recording_identity_profile.py \
  tests/unit/fisheye/test_import_organized_recordings_analysis.py \
  tests/unit/fisheye/test_import_recordings_analysis.py \
  tests/unit/fisheye/test_run_recording_analysis_pipeline.py \
  tests/unit/fisheye/test_registry_shadow_publish.py \
  tests/unit/fisheye/test_registry_prune_stale_datasets.py \
  tests/unit/fisheye/test_registry_dedupe.py \
  tests/unit/fisheye/test_registry_recording_import_receipt_bindings.py \
  tests/unit/fisheye/test_registry_writer_boundary_bsub.py \
  tests/unit/fisheye/test_registry_acquisition_batches.py \
  tests/unit/fisheye/test_registry_sqlite_concurrency.py \
  tests/unit/fisheye/test_reconcile_dataset_from_root.py \
  tests/unit/fisheye/test_registry_recording_only_context.py \
  tests/unit/fisheye/test_registry_acquisition_video_streams.py \
  tests/unit/fisheye/test_subject_mask_data_profile.py \
  tests/unit/fisheye/test_recording_manifest_import_status.py \
  tests/unit/fisheye/test_backfill_subject_experiment_setup.py \
  tests/unit/fisheye/test_migrate_count_only_subject_context.py \
  tests/unit/fisheye/test_set_recording_subject_metadata.py \
  tests/unit/fisheye/test_prepare_detect_training_smoke.py \
  tests/unit/fisheye/test_registry_maintenance.py::test_recording_entity_backfill_rejects_unbound_current_profile \
  tests/unit/fisheye/test_draft_video_only_organizer_manifest.py \
  tests/unit/fisheye/test_organize_recordings_diagnostics.py \
  tests/unit/fisheye/test_organize_recordings_external_ipc.py \
  tests/unit/fisheye/test_organize_recordings_keyframe_flags.py \
  tests/unit/fisheye/test_organize_recordings_legacy_h5.py \
  tests/unit/fisheye/test_organize_recordings_video_only.py \
  tests/unit/fisheye/test_two_session_four_camera_current_authority.py -q
```

### 4.7 Post-commit inventory and ordered completion package

Six parallel read-only inventories reviewed `6969043ef801` for writer entry
points, subtraction, current-v2 safety, operational activation, plan drift, and
sequencing. They agree that Step 3 must not start yet. The Step 2 implementation
is coherent for cooperating callers, but it has not established one global
canonical-registry writer boundary and has not yet displaced enough old code to
count as consolidation.

The inventory separates four kinds of registry access that must not be treated
as one undifferentiated writer list:

| Access class | Current disposition | Required boundary |
|---|---|---|
| Current-v2 identity and locator projection | Only the organized importer and single-recording pipeline use the designated-host shadow gateway. | One authorized gateway; no generic `upsert_dataset()`, `upsert_recording()`, scan, refresh, or maintenance path may infer or mutate current-v2 identity. |
| Derived/status/profile projection | These paths generally do not own source identity, but many still open the canonical SQLite file directly and can transitively update `datasets`. | Preserve their semantic work and operation-specific callbacks; publish canonical-registry changes through the shared shadow/CAS boundary and consume verified source identity without routing them through the source-identity projection API. |
| Destructive administration | Prune, dedupe, repair, and path-audit tools have different purposes and some bound-row fences, but still perform independent canonical writes. | Require explicit admin mode, durable backup, shadow publication, complete Palette-runtime integrity checks, and source/bound-row rejection. |
| Read-only query/report/model selection | Many callers instantiate `Registry(path)`, which opens a writable connection and may initialize or migrate the schema even when the operation is logically read-only. | Add/use a read-only facade or `mode=ro` plus `PRAGMA query_only=1`; readers must not create journals, run migrations, or participate in writer coordination. |

The highest-priority concrete bypasses are `registry_rescan` (whose
`--safe-shadow-publish` path calls the generic publisher without the
designated-host gateway), `reconcile_sweep`, registry scan/reconcile entry
points, `stage_complete`, `inline_refresh`, maintenance backfills, clipped
metadata repair, prune/dedupe, and several cluster, derived, and training
finalizers. Existing LSF callers of `registry_rescan --safe-shadow-publish`
include whole-video detection, native-detection campaigns, geometry approval,
and arena-geometry review. The disabled projection-refresh `--apply` submitter
does not fence these paths.

#### Issuer decision: authorization before cryptography

The receipt self-digest proves integrity, not who was authorized to publish it.
Likewise, a host name supplied through an environment variable is a routing
assertion, not authentication. Before adding another contract or signature,
write down the actual threat model:

- If the threat is accidental or concurrent jobs in a trusted lab account,
  make operating-system permissions and an access-controlled designated writer
  service the authorization boundary. Only that service may write the canonical
  registry, lock, shadow temp root, and backup root; submitted jobs carry an
  allowlisted operation/commit and the gateway records host, job, actor, and
  hashes.
- If an actor with arbitrary artifact and canonical-registry write access is in
  scope, the present design cannot authenticate authorship. That stronger model
  requires managed signing keys or an external trusted service. Do not simulate
  authentication with another self-digest or caller-provided string.

This decision is an activation gate, but cryptographic signing is not assumed
to be necessary without the stronger threat model.

#### Ordered completion packages

| Order | Package | Implementation scope | Acceptance and deletion gate |
|---:|---|---|---|
| 1 | Close the registry access boundary | Finish the generated writer/read-access census. Make apply paths targeting the durable canonical registry use one designated-host shadow publisher with operation-specific callbacks, move query-only users to read-only SQLite connections, and add a static CI ratchet for unapproved writable canonical opens. Filesystem/service authorization is an operational prerequisite; hashes and CI cannot fence an actor that independently retains canonical write permission. Quiesce or fence noncooperating writers so none can mutate the canonical file between the shadow publisher's final source hash and `os.replace()`. | Every production entry point has an explicit class and owner. Direct canonical writes outside approved gateway callbacks fail. Read-only paths run with `query_only=1`, perform no migration, and create no WAL/journal sidecars. |
| 2 | Close current-v2 publication races | Bind or recheck one stable source/consolidated metadata generation at the receipt boundary; make every mutable importer open explicitly unconsolidated; reject redirected receipt files and redirected `.imports` directories; reopen the finalized root before downstream projection; and decide/enforce whether `(session_uuid, camera_id)` is unique within the current-v2 source-recording profile. | Mutation between source reads, consolidation and receipt minting, or receipt and projection fails without authority rows. Direct and consolidated reopen agree. A second recording claim for the same session/camera fails if the current-v2 profile declares that invariant. |
| 3 | Preserve operational evidence without creating a second authority | Persist the full `RegistryShadowPublication` result for successful operations. Emit an idempotent rejected-attempt report outside the rolled-back projection transaction, bound to operation/attempt ID, claim and receipt digests when available, artifact, reason, host/job/actor, source hash, backup/runtime context, and only the state-specific candidate/staged/published fields that exist. Reports are telemetry/evidence and never select identity. | Injected conflicts leave identity/receipt/current tables unchanged while exactly one durable report survives. Successful operations retain the candidate/staged/published hash and backup evidence rather than reducing the result to `dataset_id`. Rejected reports never fabricate later-phase fields. |
| 4 | Route, then subtract current identity implementations | Remove current-profile identity inference and mutation only after all supported current callers use the gateway or verified reader. Split maintenance artifact inventory from identity SQL; retain explicit unprofiled/derived compatibility behind one named boundary. Do not delete generic dataset/recording registration wholesale because it still serves derived and historical artifacts. | No current-v2 caller reaches legacy precedence or direct identity SQL. Delete the maintenance `recordings` upsert and `datasets.recording_id` update (about 86 production lines), plus approximately 60–95 lines of normal current-profile fallback/call-site plumbing. Report actual diff, not estimates. |
| 5 | Compact the new implementation | Remove only demonstrated duplicate work and dead API surface; keep the five migration-73 tables, append-only/duplicate-insert triggers, receipt/source rereads, transaction fences, sidecar checks, and consolidated-publication verification unless an equivalent stronger boundary replaces them. | The current-v2 campaign slice becomes materially smaller and has one implementation per guarantee. The unrelated approximately 258-line dead refined-keypoint status subsystem is a separate Step 4 deletion and must not disguise Step 2 accounting. |
| 6 | Prove the final reduced boundary | After packages 1–5, rerun focused tests and use a clean commit-pinned deployment for a quarantined physical canary with real media probing, frame clock, Git identity, consolidated reopen, four cameras sharing one session, receipt binding, verified read, and failure injection. The canary uses scratch Zarr and registry paths and is selector-ineligible. | Exact writer -> receipt -> shadow projection -> close/reopen -> verified reader succeeds with no canonical or production-selector side effect. Canary evidence applies to the code that will be reviewed, not a pre-subtraction implementation. |
| 7 | Run required CI and stage any later production rollout | Run every required workflow check. A later canonical migration/operational smoke requires a Palette-runtime backup and complete `integrity_check`/`foreign_key_check`, plus explicit approval; it is not part of this documentation update. | All required checks succeed. Until then, and until the operational gates above close, the branch remains incomplete, not merge-ready, and prohibited from shared-checkout or production activation. |

Compatibility remains deferred by decision, but current-v2 entry points must
fail before mutation rather than fall through to unprofiled precedence. The
direct `repair_recording_identities.py` mutator remains unsupported until it is
retired or redesigned around a future correction revision. Its deletion would
be useful cleanup, but it would not prove that the current identity writer is
complete.

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

Inventory active workload producers as a separate, execution-facing slice
generated from the executable producer/profile declarations used by planning.
For every producer reachable from a maintained workflow catalog, operation
builder, submission script, or CLI, record the generated command, exact writer
or publisher, output profile and canonical claim, full-strength resolver
branch, and real-writer-to-unpatched-reader coverage. Completion, selector
eligibility, a schema marker, or an authority-sounding run name does not prove
that the producer emits the authority required by its downstream node. Unknown,
adapter-only, and legacy-only rows block a requested canonical workload only
when reachable in its transitive dependency closure. The repository CI ratchet
separately rejects newly added maintained production entry points that lack a
declaration and boundary-test disposition.

The recording-identity slice is implemented at `d816771d11cb`. It inventories
the registry schema globally but applies row-level artifact findings only to
the declared `explicit_source_layout` scope. The result closes the bounded
identity-evidence subtask; it does not close writer-generation coverage,
unmarked-artifact reconciliation, compatibility repair, or the remaining
fact-family census. Its observation cap is per metadata scan scope and never a
frame-row limit; a reached cap makes coverage incomplete and the command
non-successful.

**Gate:** for the identity slice, the report is read-only, emits no effective
identity or repair, records capped evidence as incomplete/non-success with
severity determined by cap type, and is bound to the registry snapshot it
inspected. For every other slice, no proposed deletion or new authority
proceeds without an explicit disposition and a named consumer migration.
Planning remains read-only and may always report a DAG. A node is not reusable,
runnable, or submittable until every concrete required input names a supported
producer profile or exact successor and passes the shared consumer resolver.
Outputs not yet produced remain pending real publication receipts; planning
does not invent their artifact digests or authority records.

### Step 2 — Consolidate recording identity, including `datasets`

Land the read-only evidence resolver and one registry projection writer, then
route normal registration and maintenance through it and normalize
profile/status joins. Add the source-import receipt that binds the exact
producer commit, dirty state, configuration, source evidence, and identity
decision. The initial current-v2 implementation is immutable and rejects
correction. An explicit correction revision is a future campaign contract, not
an activation requirement for that deliberately narrow initial profile. A
durable rejected-attempt report is an activation requirement because it closes
the operational audit trail without becoming another identity authority. Do
not use the selected existing corpus or downstream run provenance as a
current-writer oracle.

**Gate:** `recording_id` and `session_uuid` remain separate; ordinary backfill
cannot change a non-null identity or erase one with null; unsupported correction
and relocation fail closed; and sidecar, donor, camera, and frame-map conflicts
fail before output. If correction is later enabled, it must be versioned,
audited, and compare-and-swap guarded. Current synthetic writer-to-verified-
reader round trips pass, followed by a commit-pinned quarantined source canary
held outside production locators and registry authority. Clipped publication
has its own later profile gate. Superseded SQL and precedence branches are
deleted only after all supported current callers migrate. Existing-artifact
repair remains out of scope.

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

Include both parentless `is_run_complete(run)` and
`mark_run_complete(..., parent_group=None)` calls in the migration census.
Without the parent selector/generation context, checks inherit legacy
acceptance, and completion currently skips the strict provenance gate at
`shared/zarr_run_completion.py:197-209`. Parent scope controls selector and
generation validation; its absence must not waive producer-provenance
validation, and merely supplying a legacy parent is insufficient: strict mode
must require the parent's
`completion_epoch >= COMPLETION_EPOCH_REQUIRE_PROVENANCE` (currently 2).
Authoritative readers separately set `legacy_default=False`; that reader guard
is not a substitute for the parent completion epoch. Missing or invalid
provenance must fail the normal `mark_run_complete()` path. Existing explicit
`allow_missing_run_provenance=True` callers are temporary compatibility or
maintenance boundaries: migrate them to a named non-complete,
non-authoritative lifecycle state that cannot become selector-eligible, and
that `is_run_complete()` never treats as complete.

The explicit-bypass disposition must include completion-epoch backfill,
training-review refresh, detection-snapshot publication, and provider-position
comparison. Their maintenance or candidate purpose may remain, but normal
`complete` cannot continue to encode a weaker guarantee for those callers.

A bounded Goodbat review illustrates the gap without overstating impact: 356
runs had `run_provenance.git_sha=null`; 355 were marked complete and one
failed. All were selector-ineligible, no parent selector referenced them, and
none was authoritative. The fix is still required because `complete` must not
silently mean different provenance guarantees according to whether the caller
supplied a parent group.

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
allowed vocabulary and the same writer. Missing or bypassed run provenance is
a first-class state, cannot become selector-eligible through fallback, and
cannot be presented as authoritative completion.

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
resolver rejects. The selected run must have parent-scoped completion under
strict semantics, selector eligibility, and valid producer provenance;
`complete` alone is insufficient. Retry is idempotent and partial failure
remains visible.

### Step 6 — Unify arena geometry, calibration, and crop authority consumption

Route traditional detection, subject segmentation, arena assignment, and
registry calibration status through the selected geometry/calibration
authorities instead of legacy `dish_mask` or independent scalar ladders. Land
the crop resolver work from the crop audit: multiple supported crop profiles,
one position-authority interface, full validation per branch, and no
process-global monkey-patch adapter.

Run the global producer-inventory hardening and the targeted Sleepyfish repair
in parallel. Before a particular clipped analytics node can execute, complete
the Step 1 inventory and dynamic artifact admission for that node's transitive
dependency closure; unrelated unknown catalog entries do not block the chain.
The 2026-08-24 Sleepyfish recovery demonstrated why this local gate is required:
`finalize_keypoint_shards` produced a valid complete ordinary keypoint run, but
the workload treated it as canonical geometry authority without a canonical
coordinate manifest or successor proof. The generic planner then submitted a
bundle-backed subject-shape source without understanding that its maintained v5
profile was candidate-only; the real v003 workload failed there before eye
angles ran. The next reachable eye-angle boundary still lacks a normalized
indirect assignment-keypoint resolver. The existing keypoint successor
publisher also cannot consume the four ordinary shard-finalizer runs, and a new
keypoint publication cannot retarget already-sealed bundle/subject-shape
assignments without immutable rebinding. The exact evidence and phased repair
are recorded in
[`clipped_eye_assignment_authority_failure_2026-08-25.md`](clipped_eye_assignment_authority_failure_2026-08-25.md).

Recovery begins with a read-only proof-sufficiency result. If any required
pixel-provider, crop, frame, row, model, preprocessing, or scientific-array
claim is absent or conflicting, the result is `unmigratable` and the admitted
upstream producer must rerun. If the proof closes, the preferred route is a
general direct-hybrid terminal-evidence profile feeding the maintained strict-
v2 finalizer, not a Sleepyfish-specific migration publisher. The existing 220
shards are correctly labeled `legacy_noncanonical`, but none names the crop-
pixel work-package manifest required by the current terminal-receipt profile,
so that current profile cannot consume them unchanged.

The selected arena contract deliberately does not write the legacy `dish_mask`
projection. Production readers cannot be switched by merely changing an attr
name: first give each reader a profile-aware selected-geometry path (or an
explicit checked compatibility projection), then remove the direct legacy
read. Likewise, registry calibration must call the full
`load_selected_calibration_snapshot()` path; its current scalar ladder can
report a present group as `ok` even when `usable_camera_scale` is false.

**Gate:** each supported profile in the requested closure has a real-writer-to-
unpatched-reader CI test; the registry cannot mark calibration usable when the
canonical selected-calibration loader rejects reciprocity or manifest binding,
and status `ok` alone is never treated as proof of a usable scale. A canonical
downstream plan
must also bind the exact producer profile, artifact/run, authority proof mode,
and authority digest for every existing indirect dependency. An output not yet
published remains pending its declared receipt rather than receiving predicted
evidence. An unsatisfied assignment, coordinate, temporal, or successor
boundary leaves the node blocked during metadata preflight, before payload
scans, scratch creation, submission, or publication.

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

The current checkpoint has not reduced repository size. The Step 1 census
commit added 3,082 lines of diagnostic source, 763 focused-test lines, and a
362-line evidence document: 4,207 lines total. The diagnostic is not imported
by production modules and creates no runtime authority, but it is still real
maintenance surface. This additive checkpoint is justified only if Step 2 uses
its evidence to delete the competing writers and precedence paths. Report
runtime production, diagnostic/maintenance, tests, and docs separately so a
large diagnostic cannot be hidden inside a claimed net reduction.

| Step | Small canonical addition | Subtraction it must unlock |
|---:|---|---|
| 1 | Taxonomy and a read-only generated census; the recording-identity slice is now implemented with no runtime authority. | It does not itself subtract code. It supplies the disposition evidence needed for Step 2 and prevents duplicate inventories from being maintained by hand. |
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

- runtime production lines added and removed;
- diagnostic and maintenance lines added and removed;
- test and documentation lines added and removed;
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
5. **Compatibility scope is explicit.** Historical compatibility debt may be
   deferred, but it cannot silently feed current authority. Each currently
   supported grammar has a real-writer-to-unpatched-reader test. An adapter is
   not evidence that a profile is fully supported, and an unmarked artifact is
   not labeled legacy merely because it falls outside a census cohort.
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
11. **Producer scopes do not substitute for one another.** Downstream run or
    stage provenance cannot backfill source-root importer provenance. Missing,
    unknown, and explicitly bypassed producer evidence remain first-class
    states.
12. **Current-writer proof is generated deliberately.** Schema/layout markers
    and an old corpus do not identify a writer revision. Acceptance uses
    synthetic writer-to-unpatched-reader tests and a commit-pinned source
    canary held outside production locators and authority before activation.
13. **Bounded evidence fails closed.** Observation, metadata-node, or identity-
    cardinality caps make a scope incomplete and produce a non-success result.
    Large frame-index row counts remain valid and are not treated as caps.
14. **Producer admission precedes consumer execution, not read-only planning.**
    Every active workload node names an executable producer/profile declaration
    and resolves concrete direct and indirect authorities through shared full-
    strength interfaces. Future upstream outputs remain `pending_receipt` until
    their real publication receipts resolve. Completion, eligibility, paths,
    names, and schema markers are supporting evidence, not substitutes for the
    declared authority proof. Unknown or unsatisfied boundaries produce typed
    blocked nodes and cannot create scratch state or be submitted.

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
- It does not repair or classify the unmarked Batman and Goodbatbadbat corpus,
  reconstruct missing historical importer commits, or require the entire old
  corpus to satisfy current conventions. Those are explicit deferred
  compatibility tasks. Current implementations must still reject unresolved
  identity/provenance rather than interpreting deferred artifacts by fallback.
- It does not authorize `registry/repair_recording_identities.py` to mutate the
  observed conflicts. That tool remains deferred until it is routed through
  the versioned correction and projection-writer boundary or retired.

## 10. Source map

The main governing contracts and implementations for this plan are:

- Implemented recording-identity evidence baseline:
  [`recording_identity_census_2026-08-25.md`](recording_identity_census_2026-08-25.md),
  `registry/recording_identity_census.py:1-3082`, and
  `tests/unit/fisheye/test_recording_identity_census.py`.
- Committed current-v2 identity implementation at `6969043ef801`:
  `shared/source_recording_identity.py`,
  `shared/recording_import_receipt.py`,
  `registry/recording_identity_authority.py`,
  `registry/shadow_publish.py`, and migration 73 at
  `registry/migration_bodies.py:8650-9122`.
- Activation-blocking registry writer surfaces:
  `utils/registry_rescan.py`, `registry/reconcile_sweep.py`,
  `registry/scan.py`, `registry/stage_complete.py`,
  `registry/inline_refresh.py`, `registry/maintenance.py`,
  `registry/prune_stale_datasets.py`, `registry/dedupe.py`, and
  `utils/backfill_clipped_analysis_metadata.py`. These are different
  functional classes; the required common property is that none may mutate the
  durable canonical SQLite file outside the approved publication boundary.
- Registry authority and immutable identity policy:
  [`registry_data_governance_policy.md`](../registry_data_governance_policy.md),
  `registry/db.py:618-624,742-818,2554-2557,2621-2640`, and
  `registry/maintenance.py:785-948`, with joined-view identity sources at
  `registry/migration_bodies.py:4430-4477,6580-6582,6760-6764`; the independent
  direct mutator requiring disposition is
  `registry/repair_recording_identities.py:1-6,132-183,214-274`.
- Identity normalization history:
  [`recording_registry_normalization_todo.md`](../recording_registry_normalization_todo.md)
  and
  [`acquisition_batch_registry_contract.md`](../acquisition_batch_registry_contract.md).
- Source-import versus downstream producer provenance:
  `utils/import_recording_analysis.py:225-342`,
  `shared/import_video_metadata.py:393-535`,
  `shared/run_provenance.py:114-133,241-289,405-435`, and
  `shared/stage_provenance.py:136-201`.
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
  `shared/zarr_run_completion.py:107-116,197-231,258-307,397-443,506-681`, and
  `registry/maintenance.py:4151-4188,4218-4248`; representative parentless
  completion/provenance call sites are
  `analysis_workflows/materializers/subject_position.py:399-410,811-816`,
  `analysis_workflows/materializers/provider_epoch_behavior_summary.py:955-975,1138-1145`,
  and
  `analysis_workflows/materializers/provider_track_motion.py:1012-1032,1497-1502`.
- Explicit missing-provenance bypasses requiring non-complete disposition:
  `utils/backfill_completion_epoch.py:361-368`,
  `utils/refresh_training_review_status.py:281-288`,
  `shared/zarr/detection_snapshot_publication.py:627-635,967-975`, and
  `analysis_workflows/materializers/provider_position_comparison.py:507-513,540-548`.
- Zarr metadata lifecycle:
  `shared/zarr_io.py:14-48`, `registry/db.py:218-228`,
  `registry/maintenance.py:141-151`, `registry/chaser_metadata.py:85-90`, and
  `registry/stimulus_metadata_backfill.py:80-85`, plus the compatibility opener
  in `shared/zarr_helpers.py:413-425` and its generic caller at
  `shared/recording.py:167-174`.
- Recording-step status and completion:
  `registry/status_ledger.py:11-12,105-244`,
  `registry/stage_complete.py:488-495`,
  `registry/maintenance.py:5357-5386,6971-7060`, and
  `registry/migration_bodies.py:3242-3261`.
- Existing serialized-stage finalizer and detection coverage gap:
  `analysis_workflows/registry_finalize.py:1-121,310-633` and
  `analysis_workflows/storage_contract_catalog.py:227-451`.
- Registry scientific-projection refresh paths:
  `registry/db.py:5347,6010,6892,7127`,
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
  `registry/maintenance.py:4023-4108`.
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
