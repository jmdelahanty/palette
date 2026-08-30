# Artifact Identity, Merkle, and Receipt Architecture Review — 2026-08-29

<!-- contract-meta
status: diagnostic_snapshot
code_revision: 4326af6d
branch: agent/palette/recording-identity-evidence-20260825
authority: evidence and recommendations only; not a runtime contract
-->

**Date:** 2026-08-29
**Scope:** Read-only review of artifact identity, content hashing, provenance,
validation receipts, atomic publication, selector authority, Zarr metadata, and
scientific-validation boundaries.
**Working-tree note:** The checkout was already dirty when this review began.
Claims from modified or untracked diagnostics are treated as proposals, not as
durable repository contracts. No production data, registry state, selectors,
or publication state were changed.

## Evidence notation

- **[C]** — verified directly in committed or tracked implementation and tests.
- **[V]** — verified by a read-only command during this review.
- **[W]** — present in the current dirty working tree; not necessarily committed
  or accepted.
- **[A]** — code-reading result independently reported by one or more parallel
  audit lanes and reconciled against repository structure.

Line numbers are as of the working tree at revision `4326af6d`; later edits may
move them.

---

## Executive conclusion

Palette's conceptual model is stronger than its implementation packaging.
Several modern paths already understand that byte integrity, scientific
validity, publication authority, and authenticated authorship are different
claims. Atomic publication and stage-specific semantic validation solve real
distributed-correctness problems and should remain.

The principal weakness is fragmentation:

- there is no repository-wide, location-independent artifact fingerprint;
- several incompatible digest and provenance grammars coexist;
- some publication paths persist nested or duplicative receipt envelopes;
- some artifacts are scanned repeatedly despite persisted evidence;
- identity fields sometimes mix scientific content with paths, selectors, or
  other lifecycle state; and
- higher-level source-selection and semantic gates remain weaker than the
  downstream hash machinery.

The practical target is:

> **one stage-facing artifact reference, one common immutable manifest
> envelope, several explicitly scoped roots, strong semantic and atomic gates,
> and no security machinery without a security threat model.**

This is close to the proposed "one immutable manifest plus one artifact
fingerprint" architecture, with one important qualification: one external
handle should not mean one undifferentiated hash for logical content,
provenance, physical storage, validation, and publication state.

---

## Recommended identity model

| Record | Claim | Stable across |
|---|---|---|
| `logical_content_root` | Decoded scientific values and logical schema | codec, chunking, layout, and relocation changes |
| `artifact_manifest_fingerprint` | Producer, exact code/config, exact input artifact references, schema, and logical outputs | relocation and publication-state changes |
| `physical_payload_root` | Exact encoded bytes/files/shards | transfer and replica checks only; not re-encoding |
| `validation_attestation` | Validator identity, version, policy, report digest, result, and bound artifact | relocation; may change when validation is rerun |
| `publication_record` | Locator, owner, generation, selector, consolidation generation, and lifecycle result | nothing; this is operational event evidence |
| `manifest_document_sha256` | Exact serialized envelope identity | nothing; this is a document-corruption check |

Downstream stages should normally pass only a reference such as:

```text
artifact:sha256:<artifact_manifest_fingerprint>
```

A resolver maps that identity to one or more locators. Paths, hosts, timestamps,
leases, scheduler IDs, selector state, and mutable telemetry do not belong in
the artifact fingerprint.

Editable authorities require an additional rule: accepted edits must either
create immutable revisions or carry explicit revision and row-source identity.
The canonical artifact fingerprint is minted when an immutable snapshot is
sealed or promoted, not while the dense editable surface is still mutable.

---

## Where the repository already adheres well

### 1. Integrity, science, publication, and authorship are separated

**[C]** `docs/zarr_payload_validation_receipt_contract.md:42-55` separates:

1. construction evidence;
2. byte-integrity evidence;
3. scientific-validation evidence; and
4. publication/selector authority.

It explicitly states that SHA-256 content binding does not establish scientific
correctness or authenticated authorship.

**[C]** `src/fisheye/shared/zarr_payload_receipt.py:519-549` labels the payload
integrity record as immutable-output integrity rather than scientific
validation. `:632-691` separately binds validator identity, numerical policy,
scientific-manifest digest, and result to the integrity roots.

**[C]** The track receipt canary is particularly strong evidence. The decoded
root and scientific totals matched, yet independent publication validation
rejected the shortcut because it did not establish row identity, temporal
lineage, source authority, domains, aliases, and derivations. See
`docs/diagnostics/track_manifest_receipt_canary_2026-07-25.md:43-55`.

### 2. Atomic publication is real correctness machinery

**[C]** `src/fisheye/shared/atomic_run_publisher.py` owns crash and concurrency
mechanics rather than scientific claims. Its maintained path performs hidden
same-parent staging, physical copy verification, ownership and ineligibility
stamping, validation, `os.replace`, fresh reopen checks, completion, guarded
selector activation, rollback, and immutable tombstoning. Relevant seams are
`:788-815`, `:888-1074`, and `:1075-1231`.

**[C]** `src/fisheye/shared/selector_activation.py:309-566` freshly resolves the
canonical parent, checks owner and generation state, validates the intended
child, and writes eligibility last. Hostile concurrency and interruption paths
are covered in `tests/unit/fisheye/test_atomic_run_publisher.py` and
`tests/unit/fisheye/test_selector_activation.py`.

This machinery is foundational. A manifest fingerprint cannot replace atomic
rename, writer exclusion, or selector compare-and-swap.

### 3. Modern manifests already approximate the target

**[C]** Canonical detection manifests bind producer/model/source evidence,
logical schema, storage declarations, and exact per-array shape, dtype, and
decoded-content hashes in
`src/fisheye/shared/zarr/canonical_detection_manifest.py:119-150` and
`:600-679`.

**[C]** Tracking manifests bind configuration, provenance, source identity,
output-array declarations, and a payload digest in
`src/fisheye/tracking/run_manifest.py:142-184`.

**[C]** Keypoint activation binds both full upstream manifest digests and
narrower logical-content digests, then rechecks metadata and decoded arrays
before authority changes in
`src/fisheye/shared/zarr/keypoint_bundle_activation.py:252-399`.

**[C]** Analytics export is the cleanest existing publication model:
`src/fisheye/analytics_exports/publication.py` validates an immutable
generation and uses a manifest rename as the visibility commit.

### 4. Meaningful semantic validators exist

**[C]** Subject-mask validation rederives thresholded masks, area, centroids,
bounding boxes, presence, dense-cache agreement, identity, frame offsets, and
source-row bindings from authoritative arrays in
`src/fisheye/shared/zarr/subject_mask_schema.py:374-625`.

**[C]** Body-frame validation checks unit and orthogonal axes, handedness,
heading derivation, valid/NaN semantics, source-row identity, and signatures in
`src/fisheye/shared/zarr/body_frame_schema.py:209-447`.

**[C]** Gaze validation explicitly acknowledges that numerical identities
cannot prove the biological assumption that a directionless ellipse axis is a
directed gaze vector; review remains required. See
`src/fisheye/analysis/gaze_convention_validation.py:752-773`.

These controls operate above byte integrity. They should receive at least as
much engineering attention as content-addressing machinery.

### 5. Some distributed receipts prove real composition invariants

**[C]** Subject-mask row-unit receipts are stable across worker batch sizes and
enforce complete, ordered, gap-free coverage in
`src/fisheye/shared/zarr/subject_mask_validation_receipt.py:51-177` and
`:520-580`.

**[C]** Sharded copy assigns workers non-overlapping physical shards, hashes
decoded bytes, writes, rereads, and verifies them in
`src/fisheye/shared/zarr_sharded_copy.py:300-430`.

These are legitimate distributed-data guarantees. Receipt simplification must
preserve partition coverage, row identity, and final-layout verification.

---

## Current Merkle assessment

Palette does not presently implement the partial-verification Merkle system
that the name may suggest.

**[C]** `src/fisheye/shared/zarr_payload_receipt.py:72-111` builds every physical
leaf as `{relative_path, size_bytes, sha256}` and computes:

```text
root_sha256 = SHA256(canonical_json(complete_sorted_leaf_list))
```

Decoded array/run roots and immutable-metadata roots similarly hash complete
child lists (`:114-140`, `:217-277`).

**[C]** `verify_payload_integrity_receipt` reconstructs and rereads the complete
physical record when physical verification is enabled (`:601-629`). There is no
inclusion-proof generator, sibling path, or selected-chunk proof verifier.

Consequences:

- a reader cannot verify chunks 320-340 with an `O(log n)` portable proof;
- physical leaves identify files or shards, not necessarily logical Zarr chunks;
- physical identity changes under re-encoding or layout changes; and
- ordinary verification remains a full enumeration and scan.

The present roots remain useful as deterministic collection seals for full
artifact validation, transfer comparison, forensic diffing, or scheduled
scrubbing. More precise names would be `aggregate_inventory_root` or
`sorted_leaf_inventory_root`.

Do not spread this receipt family to sibling stages on the claim that it already
supports partial verification. Either retain a flat optional leaf inventory, or
implement a domain-separated tree and proof API only after choosing a concrete
subset-read, replica-transfer, or preservation-scrub consumer.

Track kinematics is the natural pilot because narrow readers currently pay for
broad verification. Logical decoded leaves and compressed physical-object
leaves must remain distinct.

---

## Where the repository is overengineered

### 1. Repeated and nested proof envelopes

**[C]** Track publication persists a detailed scientific manifest and digest,
derives a compact publication commit from largely overlapping hashes, and also
stores payload integrity and validation/binding receipts. See
`src/fisheye/analysis/track_kinematics.py:9560-9605` and `:11791-12032`.

Some receipts measurably move work out of a publication critical section, so
the correct change is consolidation rather than deleting the scientific
validator or atomic transaction. Each retained envelope must prove a unique
claim or eliminate a measured rescan.

**[C]** Subject-mask coordinate validation embeds earlier evidence, hashes the
receipt payload, hashes the complete document again, and stores a sibling digest
in
`src/fisheye/shared/zarr/subject_mask_coordinate_validation_receipt.py:509-616`.
The source/coordinate/equivalence claims are useful; nested copies and repeated
self-digests do not create a new trust boundary.

### 2. Hash grammar fragmentation

**[C]** `src/fisheye/shared/zarr/manifest_digest.py:18-33` provides a strict
canonical JSON grammar: sorted keys, compact UTF-8, and `allow_nan=False`.

Other modules duplicate or vary that grammar. Examples include ASCII escaping
in `src/fisheye/group_statistics/legacy_arrow.py:67-75`, default JSON NaN
behavior in `src/fisheye/analysis/chaser_profiles.py:106-108`, and a duplicate
strict implementation in `src/fisheye/shared/zarr_payload_receipt.py:37-48`.

Array digests also differ in whether dtype and shape are included in the
preimage. Legacy digests must remain interpretable, but new schemas need one
versioned algorithm registry and explicit domain separators.

### 3. Ambiguous identity names and scopes

**[C]** `run_lineage_fingerprint` intentionally excludes output content and
operational details, yet writes one value under `source_fingerprint`,
`source_lineage_hash`, and `lineage_hash` in
`src/fisheye/shared/run_lineage_fingerprint.py:300-387`.

**[C]** Some manifest payloads include run names, paths, or eligibility state.
For example, tracking includes `run_name`, `run_path`, and publication state in
`src/fisheye/tracking/run_manifest.py:142-184`. Identical immutable content can
therefore acquire a different identity after relocation or lifecycle changes.

The repository should reserve unqualified `artifact_fingerprint` for a portable
immutable artifact identity and use narrower names for task equivalence,
logical content, exact document bytes, and publication events.

### 4. Operational evidence with no consumer

**[C]** Atomic publication writes `cluster_output_staging` through multiple
phases in `src/fisheye/shared/atomic_run_publisher.py:975-1042`, but the record
is mutable and commonly excluded from scientific manifests. Consumers cannot
treat it as immutable evidence, so it often coexists with another scan.

Either:

- make it explicitly best-effort telemetry and stop relying on it for claims; or
- emit one bounded immutable evidence object referenced by digest from the
  publication record.

Do not add another receipt merely to authenticate the existing receipt.

---

## Higher-priority best-practice gaps

### 1. Weak root input identity

**[C]** Source-video identity remains `stat_v1`: path, size, and mtime rather
than content bytes in `src/fisheye/shared/import_source_fingerprint.py:35-70`.

**[C]** `fingerprint_artifact` may reuse a registry or sidecar digest when size
and mtime match and is explicitly best-effort in
`src/fisheye/shared/artifact_fingerprint.py:108-178`. A separate
`require_artifact_content_identity` helper correctly performs direct rehashing
at scientific commit boundaries (`:181-214`), but use is not universal.

**[A]** A current detection path can describe the cached model fingerprint as
verified in `src/fisheye/detection/detect_yolo.py:2426`. A same-size model
replacement with a restored mtime and stale sidecar can therefore retain the
expected identity. Authoritative model and source boundaries should directly
rehash bytes or use a separately trusted immutable acquisition identifier.

### 2. Provenance is fragmented and shallowly validated

**[C]** `run_provenance` and stage provenance use competing schemas for code,
parameters, inputs, and artifacts. Generic run-provenance validation chiefly
requires nonempty `git_sha` and `config_hash`; it does not recompute the config
hash, validate an exact schema identity, or require content-bound input
artifacts. See `src/fisheye/shared/run_provenance.py:397-435`.

Dirty code is recorded as a boolean/list rather than an exact source-tree or
diff digest (`:114-132`). Selector-eligible production should require either a
clean commit or a reproducible source-tree/diff identity.

### 3. Some semantic and source-selection gates remain weaker than the hashes

**[A]** Refined-mask promotion uses a validator that checks structure and
decodability but does not rederive every persisted area/centroid/bounding-box
value from authoritative dense masks. See
`src/fisheye/utils/validate_refined_subject_mask_contract.py:626-810` and the
promotion entry point in
`src/fisheye/utils/promote_refined_subject_mask_run.py:358-430`.

**[A]** The subject-mask training exporter can select a requested, `latest`, or
lexicographically last run without uniformly requiring completion, eligibility,
and a validated source-manifest fingerprint in
`src/fisheye/utils/export_subject_mask_training_zarr.py:131-218`.

These are concrete routes to producing a rigorously encoded or hashed version
of the wrong scientific input.

### 4. Publication lifecycle rules are not uniform

**[C]** `src/fisheye/shared/zarr_run_completion.py:427-443` treats a missing
`stage_selector_eligible` marker as eligible for compatibility, contrary to the
strict fail-closed direction for new publications.

**[A]** Fresh materialization and candidate-promotion paths for some analytics
families differ in whether consolidated metadata is rebuilt and verified before
reporting promotion. Publication semantics should depend on artifact lifecycle,
not which entry point produced the same family.

**[V]** The required static metadata-mode ratchet currently fails:

```text
$ scripts/py scripts/check_zarr_open_group_modes.py --no-update-on-shrink
Zarr open-group metadata-mode ratchet failed; new or modified calls must pass
use_consolidated explicitly:
  src/fisheye/utils/materialize_clipped_keypoint_direct_hybrid_terminal.py:243
```

This checkout is therefore not CI-green or merge-ready on current evidence.

### 5. The threat model is not yet a durable repository-wide contract

**[C]** Committed documentation correctly says self-digests provide content
binding, not authenticated authorship, and treats signing as optional for a real
chain-of-custody requirement.

**[W]** The strongest explicit trusted-lab-versus-malicious-writer analysis is
currently in the modified
`docs/diagnostics/source_of_truth_consolidation_plan_2026-08-25.md:650-669`.
It correctly recommends OS permissions and one designated writer for the normal
cooperating-job threat model, and managed keys or an external service only if an
actor with arbitrary artifact and registry write access is in scope. Because
that text is uncommitted, it should be ratified separately rather than cited as
established policy.

There is no artifact-signing key infrastructure or transparency-log trust
anchor in the repository. Do not simulate either with another colocated
self-digest.

---

## Recommended migration order

### P0 — Fix authority roots before adding receipts

1. Directly hash authoritative model files at load and publication boundaries.
2. Replace `stat_v1` source-video identity with a content digest computed while
   bytes are already streamed, or with a separately trusted immutable
   acquisition identity.
3. Require completed, eligible, manifest-bound sources in training/export paths.
4. Require the strongest available semantic validator before refined-mask
   promotion.
5. Resolve the explicit Zarr metadata-mode CI failure.

### P1 — Introduce one common identity envelope

1. Define `palette.artifact_manifest.v1` with a strict shared canonicalizer.
2. Keep stage-specific semantic payloads and validators inside or referenced by
   that envelope.
3. Define a location-independent `artifact_manifest_fingerprint` over exact
   producer/code/config, exact upstream artifact references, logical schema,
   and logical output roots.
4. Add registry resolution from artifact fingerprint to one or more locators.
5. Dual-write beside existing manifests in one modern family before migration;
   tracking is the natural first candidate.

### P1 — Consolidate evidence without deleting guarantees

1. Inventory each receipt by exact claim and consumer.
2. Retain a receipt only if it proves a distinct actor/phase boundary, preserves
   distributed coverage or lineage, or measurably eliminates a rescan.
3. Replace nested receipt copies with schema-and-digest references to immutable
   evidence documents.
4. Keep mutable selector, lease, path, host, timing, and scheduler data outside
   artifact identity.
5. Preserve atomic publication, semantic rederivation, row identity, partition
   coverage, and fail-closed selector checks.

### P2 — Decide the physical-root consumer

Choose one:

- scheduled bit-rot scrubbing;
- replica comparison and repair;
- transfer verification; or
- partial-read proof verification.

If none is selected, retain at most a simple optional inventory root and stop
expanding physical receipt machinery. If partial verification is selected,
implement stable logical-chunk leaves, explicit domain separation, proof
generation/verification, and an independently obtained trusted root. Keep large
leaf inventories outside compact Zarr attributes.

### P2 — Ratify the threat model

State the normal trust boundary explicitly: cooperating jobs under trusted lab
accounts, accidental corruption, crashes, and concurrent writers. Under that
model, designated-writer and filesystem/registry access controls are the trust
anchor.

If arbitrary storage writers or regulated nonrepudiation later enter scope,
sign one canonical artifact root using managed isolated keys and optionally
anchor it externally. Do not sign every nested receipt.

---

## Parallel subtraction audit — concrete removal queue

Ten read-only audit lanes traced producers, consumers, tests, and compatibility
requirements for the machinery discussed above. The queue uses four
dispositions:

- **Delete now** — no post-transaction production consumer, or an exact local
  replacement already exists. Historical data may remain readable.
- **Stop new writes** — retain a legacy read adapter, but new artifacts should
  no longer emit the duplicated field or record.
- **Migrate first** — a schema, archive, or consumer migration must precede
  deletion.
- **Keep** — the mechanism protects a distinct correctness boundary.

The line-level findings below are **[A]** code-reading evidence from the parallel
lanes unless marked otherwise. Before implementation, refresh line numbers and
rerun the listed reference searches because this working tree is active.

### Ranked summary

| ID | Disposition | Candidate | Proposed survivor |
|---|---|---|---|
| S1 | Delete now | Persisted track binding-validation receipt | Transaction result plus final manifest/attestation |
| S2 | Delete now | Direct `latest` writes immediately before `mark_run_complete` | `mark_run_complete` |
| S3 | Delete now | `mark_run_pending` and several uncalled helpers | `note_pending_latest` and typed resolver |
| S4 | Delete after API check | Raw-keypoint successor publication receipt surface | Existing preparation/reconciliation primitives or no public wrapper |
| S5 | Stop new writes | Three aliases for one run-lineage digest | One `lineage_hash` plus conflict-detecting legacy reader |
| S6 | Stop new writes | Duplicate `run_provenance` / `cli_provenance` attrs | `run_provenance` plus legacy read fallback |
| M1 | Migrate first | `track_motion_publication_commit` | Full track manifest plus canonical digest |
| M2 | Migrate first | Verbose payload-receipt envelopes and persisted flat leaves | Compact roots plus manifest validation section |
| M3 | Migrate first | Repeated track physical/metadata/decoded scans | Scoped validation at each publication boundary |
| M4 | Migrate first | `cluster_output_staging` as activation authority | Minimal immutable publication evidence plus telemetry |
| M5 | Migrate first | Subject-mask nested worker/core/bundle/import receipts | One producer manifest, member artifact refs, validation attestation |
| M6 | Migrate first | Separate subject-mask coordinate receipt and successor authority | One successor manifest with validation section |
| M7 | Migrate first | Native detection candidate and shadow receipts | Canonical manifest, completion-last commit, optional telemetry report |
| M8 | Migrate first | Competing canonical JSON/array/file/tree helpers | Versioned shared digest primitives and legacy adapters |
| M9 | Migrate first | Bespoke publishers, selector engines, and tombstone implementations | Shared transaction, selector, and failure-transition kernels |
| M10 | Migrate first | Keypoint/crop duplicate semantic attrs, ancestry chains, and digest-only “signed” wrappers | Strict manifests, parent link, content-bound terminology |

---

### S1 — Delete the persisted track binding-validation receipt

**Evidence.** The binder performs the real exhaustive binding validation at
`src/fisheye/analysis/track_kinematics.py:12275-12293`. It then:

1. constructs a generic validation receipt at `:12294-12312`;
2. verifies and persists it at `:12445-12473`; and
3. has the materializer immediately compare the persisted digest at
   `src/fisheye/analysis_workflows/materializers/track_kinematics.py:788-805`.

Repository-wide reference search found no later production consumer of
`TRACK_KINEMATICS_BINDING_VALIDATION_RECEIPT_ATTR`.

**Remove.** Stop writing the attribute, remove
`binding_validation_receipt_sha256`, remove the build/verify round trip, and
remove the materializer echo check.

**Keep.** Preserve exhaustive source-row, temporal-lineage, coordinate,
physical-scaling, and semantic validation; the post-binding payload check; proof
closure inside the rollback boundary; and the returned binding manifest digest.

**Tests.** Update
`tests/unit/fisheye/test_track_kinematics_coordinate_contract.py:775-884` and
`tests/unit/fisheye/test_track_kinematics_materializer.py:260-320,648-670` to
assert the surviving validation and rollback behavior without requiring the
attribute.

### S2 — Remove early direct selector writes

The following writers assign `latest` immediately before calling
`mark_run_complete`, which already publishes `latest_complete` and `latest` at
`src/fisheye/shared/zarr_run_completion.py:297-306`:

- `src/fisheye/utils/detection_profile.py:967`;
- `src/fisheye/utils/keypoint_profile.py:900`; and
- `src/fisheye/shared/subject_mask_profile.py:692`.

Delete the early writes. They add no successful-path behavior and can expose an
incomplete child to a raw-`latest` reader if completion or provenance validation
fails.

Add fault-injection tests proving a completion/provenance failure leaves the old
selector unchanged.

### S3 — Collapse trivial lifecycle helpers

`note_pending_latest` at
`src/fisheye/shared/zarr_run_completion.py:778-790` performs the useful pending
transition and repairs a stale `latest` that already points at the pending run.
`mark_run_pending` at `:793-797` is the weaker duplicate.

Repoint these callers, then delete `mark_run_pending`:

- `src/fisheye/analysis/detection_occupancy_runs.py:756`;
- `src/fisheye/analysis/chaser_distance_runs.py:1334`;
- `src/fisheye/analysis/stimulus_epoch_runs.py:333`;
- `src/fisheye/diagnostics/compare_realtime_offline_detections.py:1164`;
- `src/fisheye/tune/detect_training_promotion_backend.py:604,673`; and
- `src/fisheye/shared/zarr/schema.py:193,394`.

No repository callers were found for:

- `src/fisheye/shared/zarr/schema.py:356-447` (`add_processing_run`);
- `src/fisheye/shared/zarr/schema.py:450-472` (`get_latest_run`); or
- `src/fisheye/shared/zarr_run_completion.py:675-688`
  (`resolve_authoritative_run_group`).

Delete those after checking that they are not supported external APIs.

### S4 — Retire the raw-keypoint successor publication receipt surface

`publish_selector_ineligible_raw_keypoint_successor` and its receipt validator
and `raw_keypoint_successor_publication_receipt.json` live in
`src/fisheye/shared/zarr/keypoint_successor.py:499-735`. Repository-wide search
found no production caller; only
`tests/unit/fisheye/test_keypoint_successor.py:448-467` invokes the publication
wrapper.

If this is not an intended external API, remove the wrapper dataclass, second
publication envelope, validator, and sidecar. Keep
`TerminalKeypointInferenceBatch`, which is imported by clipped finalization, and
retain any preparation/reconciliation primitive that still has a planned
caller.

This item requires a public-API/product decision, not an archive migration.

### S5 — Stop writing three names for one lineage digest

`src/fisheye/shared/run_lineage_fingerprint.py:376-386` hashes one canonical
payload and writes the identical value as:

```text
source_fingerprint
source_lineage_hash
lineage_hash
```

Write only `lineage_hash` for new artifacts. Add one legacy resolver that reads
all historical names and fails if multiple present aliases disagree.

A second exact duplicate occurs in stimulus epochs:

- `src/fisheye/analysis_workflows/stimulus_epoch_candidate_execution.py:100-114`;
- `src/fisheye/analysis_workflows/materializers/stimulus_epochs.py:245-271`.

Both `lineage_hash` and `lineage_payload_sha256` hash the same canonical bytes.
Retain `lineage_hash`; accept and equality-check the old field on legacy reads.
At candidate-execution lines `214-218`, `validation_receipt_sha256` is likewise
assigned the same value as `temporal_axis_sha256` and should be reduced to one
name.

### S6 — Stop duplicate provenance writes

The following sites write the same document under both `run_provenance` and
`cli_provenance`:

- `src/fisheye/detection/detect_keypoints_yolo.py:2994-2995`;
- `src/fisheye/detection/detect_yolo.py:3932-3933`;
- `src/fisheye/utils/run_sam_subject_masks.py:2022-2023`;
- `src/fisheye/tracking/crop.py:2840-2841,4449-4450`;
- `src/fisheye/utils/import_refined_subject_mask_clip_packages.py:1901-1902`;
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py:1143-1144`; and
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py:1189-1190`.

Keep `run_provenance` canonical. Stop new `cli_provenance` writes, retain one
conflict-detecting legacy read fallback, and then remove duplicate CLI plumbing
such as `src/fisheye/cli/palette.py:1696`.

`src/fisheye/tracking/arena_assignment.py:937-995` is a broader example: it
copies inputs into stage provenance, mirrors them again at top level, writes
Git attrs through `write_stage_provenance`, rewrites the Git attrs directly,
then converts the same record into `run_provenance`. Lines `967-970` and
`973-974` are immediate candidates once direct legacy readers are checked. The
larger stage/run provenance merger requires a new completion schema version.

---

### M1 — Remove `track_motion_publication_commit`

`_track_motion_publication_commit` at
`src/fisheye/analysis/track_kinematics.py:9560-9605` is a deterministic, lossy
projection of the full manifest plus its digest. It is persisted next to the
manifest at `:11822-11832`, and readers merely recompute and compare the same
projection at `:11596-11672` and `:11934-11960`.

The external analytics consumer duplicates the projection in
`src/fisheye/analytics_exports/kinematics_samples.py:378-430`, verifies it at
`:648-671`, and hashes it into another source binding at `:760-780`.

**Survivor.** The full manifest and canonical manifest digest already contain
source authority, input authority, run derivation, and per-track position
derivations.

**Migration.** Add a manifest version that does not require a commit. Continue
validating v1/v2 commits on historical runs, but stop writing them for the new
version. Change analytics bindings to retain only the source manifest
fingerprint.

**Tests.** Cover old-manifest compatibility, new runs without commits, manifest
digest mismatch, and live semantic/payload tampering even when an attacker
recomputes the manifest digest.

### M2 — Shrink payload receipts and stop storing flat leaf inventories

The integrity body at `src/fisheye/shared/zarr_payload_receipt.py:534-549`
persists schema-constant prose and booleans:

- `receipt_role`;
- `merkle_composition`;
- `metadata_scope`;
- `closed_array_inventory`; and
- `closed_physical_payload_inventory`.

Move those meanings into the versioned schema specification.

The validation body at `:667-691` repeats `run_ref` and all three roots from the
integrity receipt, then hashes its already embedded numerical policy. No
production consumer reads those copied root fields outside the builder/verifier.
A compact v2 needs only:

```text
integrity_receipt_sha256
scientific_manifest
validator
numerical_policy
result
record_sha256
```

More importantly, the integrity attribute currently embeds every physical file
leaf (`:93-111`), every immutable-metadata leaf (`:114-140`), and every decoded
shard/array leaf (`:217-277`) inside the run's root `zarr.json` through
`src/fisheye/analysis/track_kinematics.py:11894-11899`. No production consumer
queries these lists.

A compact receipt can transiently validate detailed construction records and
persist only:

```text
decoded:  algorithm, array_count, decoded_bytes, root
physical: algorithm, file_count, physical_bytes, root
metadata: algorithm, file_count, root
```

If partial proofs are later required, use a real indexed sidecar rather than a
flat leaf list in root metadata. Add a test that serialized receipt size remains
constant as shard count grows.

Also rename the v2 `merkle_composition` claim to
`collection_digest_algorithm`. The present root is a canonical hash of all
leaves, not an inclusion-proof tree.

### M3 — Reduce repeated track scans with scoped validation

One track publication currently traverses the immutable metadata tree eight
times. Even `verify_physical_payload=False` still walks every `zarr.json` at
`src/fisheye/shared/zarr_payload_receipt.py:622-628`.

The repeated path includes:

1. receipt construction at
   `src/fisheye/analysis_workflows/materializers/track_kinematics.py:741`;
2. post-binding verification at
   `src/fisheye/analysis/track_kinematics.py:12450`;
3. generic pre-pointer validation at
   `src/fisheye/shared/atomic_run_publisher.py:1008`;
4. completion callback and immediate receipt checks at
   `src/fisheye/analysis/track_kinematics.py:2773-2801`;
5. generic final validation at
   `src/fisheye/shared/atomic_run_publisher.py:1020`; and
6. activation callback plus final physical verification at
   `src/fisheye/analysis/track_kinematics.py:2942-2968`.

Do not delete lifecycle boundaries. Split callback scopes instead:

- local: scientific plus structural validation;
- hidden copy: structural state plus mechanical copy proof;
- pre-completion: bound structural state;
- final: owner, completion, and selector state.

For track, keep one baseline build, one post-binder payload guard, and one final
pre-selector physical/metadata verification. Remove receipt I/O from the generic
completion callbacks. This reduces eight metadata walks to roughly three while
preserving the distinct mutation windows.

The reviewed-keypoint training publisher has a similar amplification at
`src/fisheye/training/training_review_compaction_publication.py:574-707`: each
included subtree is hashed about seven times. Keep the source digest, one local
semantic validator, and local-to-hidden whole-tree equality; remove the
immediate local subtree check, hidden deep subtree rehash, and post-rename deep
rehash in favor of a fresh owner/path reopen.

For rsync, `_copy_and_verify` hashes the entire source at
`src/fisheye/shared/atomic_run_publisher.py:265-266`, then rsync rereads source
and target with `--checksum` at `:269-290`; the standalone source digest is never
compared because target content hashing is deliberately omitted. Use
path/size inventory plus rsync checksum, or comparable source/target hashes, but
do not pay for both.

### M4 — Retire `cluster_output_staging` as authority

The shared publisher mutates one `cluster_output_staging` document three times
at `src/fisheye/shared/atomic_run_publisher.py:975-1042`. It combines runtime
telemetry, paths, host/job, copy summaries, complete parent-attribute snapshots,
and four validation results.

Track then exact-compares the entire mutable document before activation at
`src/fisheye/analysis/track_kinematics.py:2931-2941,2997-3006`, with caller
plumbing at
`src/fisheye/analysis_workflows/materializers/track_kinematics.py:888-901`.
Subject-shape and benchmark paths also consume selected fields.

Replace it with a compact immutable publication-evidence record:

```text
artifact_fingerprint
publication_owner_uuid
validator_contract + validation_report_digest
transfer_verification_algorithm + result
target_locator
lifecycle_state
record_sha256
```

Move host, scheduler, timings, phase lists, and whole parent snapshots to an
execution report. Selector activation should compare artifact fingerprint,
owner, completion, and validation digest rather than an entire telemetry blob.
Then default `persist_run_receipt=False`. Existing coverage at
`tests/unit/fisheye/test_atomic_run_publisher.py:79-99` already proves the
publisher can succeed without the persisted attribute.

### M5 — Flatten subject-mask evidence by reference

The worker/recording/core path currently nests full evidence documents:

- each worker binding embeds complete `scientific_identity`, `attempt`, and
  `worker_receipt` documents at
  `src/fisheye/shared/subject_mask_worker_receipt.py:1266-1324`;
- recording validation embeds the resulting assembly identity as
  `producer_evidence` at
  `src/fisheye/shared/zarr/subject_mask_validation_receipt.py:621-649`; and
- core publication persists that receipt again at
  `src/fisheye/shared/zarr/subject_mask_core_publication.py:2382-2408`.

Persist each immutable producer/worker evidence document once. The recording
manifest should contain ordered interval plus evidence-reference pairs, and the
validation attestation should reference the producer manifest fingerprint.

Attempts contain retry, supersession, path, and policy details at
`src/fisheye/shared/subject_mask_attempt.py:896-970`; keep them in operational
lineage rather than scientific artifact identity.

The bundle path duplicates member and compatibility evidence:

- member references at
  `src/fisheye/shared/zarr/subject_mask_bundle_publication.py:302-318`;
- `_bundle_cross_binding` at `:320-620`;
- live recomputation at `:1438-1518`; and
- atomic-import receipt digests folded into the manifest at `:672-697`.

Use member artifact fingerprints plus a compatibility-validator attestation.
Recompute compatibility at activation rather than persisting the whole derived
cross-binding. Subject-mask member imports can set `persist_run_receipt=False`
at `:842-880`; retain the returned operation receipt only for execution logs.

Move the core `write_receipt` and physical timing/sample/reopen telemetry at
`src/fisheye/shared/zarr/subject_mask_core_publication.py:2409-2473` outside the
artifact fingerprint.

### M6 — Merge subject-mask coordinate receipt and successor authority

The coordinate validation receipt repeats source, bundle, coordinate-record,
payload-equivalence, validator, and digest evidence in
`src/fisheye/shared/zarr/subject_mask_coordinate_validation_receipt.py:509-616`.
The successor authority stores substantially the same claims in
`src/fisheye/shared/zarr/coordinate_successor_authority.py:75-170`, and readers
cross-check both structures field by field in
`src/fisheye/shared/subject_mask_coordinate_publication.py:2840-2940` and
`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:5481-5585`.

Create `coordinate_successor_manifest.v2` with one validation-attestation
section. Preserve coordinate-record references, payload-equivalence evidence,
the optimized no-dense-read reader, and the exhaustive fallback.

Companion colocated `<attr>_sha256` values can then be removed for new records;
loaders can recompute the digest from the canonical document. External pointers
should still carry `record_sha256`. Generic dependencies at
`src/fisheye/shared/coordinate_record.py:126-223` require versioned migration,
so this is not a direct deletion.

The test-only `expected_record_names` alias at
`subject_mask_coordinate_validation_receipt.py:564-590` can be removed
immediately in favor of `expected_coordinate_record_names`.

### M7 — Replace detection candidate/shadow receipts with manifests and reports

The native candidate writes arrays, verifies source/destination logical hashes,
builds and persists the canonical manifest, and deeply validates it at
`src/fisheye/detection/native_canonical_candidate.py:304-376`. It then emits a
20-plus-field `native_detection_candidate_receipt.json` at `:377-404`.

The publication loader compares only a subset of manifest-derived fields at
`src/fisheye/analysis_workflows/native_canonical_detection_publication.py:327-353`,
then embeds the complete receipt again in its result at `:742-760`. A benchmark
adapter even mutates `receipt["output_path"]` after relocation at
`src/fisheye/utils/finalize_recording_canonical_detection_benchmark_adapter.py:370-373`.

Correct the candidate sequence to:

```text
write payload
build manifest
consolidate
reopen and deeply validate
commit completion last
```

Then remove the receipt as authority. Keep writes, timings, and consolidation
reports only as optional telemetry.

Canonical/refined shadow publishers similarly write
`shadow_publication_receipt.json` at
`src/fisheye/shared/zarr/canonical_detection_shadow.py:480-518` and
`src/fisheye/shared/zarr/refined_detection_shadow.py:204-223`. Diagnostic
canaries still read those files, so migrate them to the manifest plus a plainly
named benchmark report before stopping the sidecar write.

One prerequisite outranks the receipt cleanup: active callers invoke
`run_detection_local_publish`, which hard-fails unless an explicit legacy flag
is supplied at `src/fisheye/utils/run_detection_local_publish.py:543-578`.
Callers at `src/fisheye/utils/run_detect_with_registry_model.py:496-522`,
`src/fisheye/utils/run_detections_batch.py:866-887`, and
`src/fisheye/cluster/clipped_detection.py:469-504` omit that flag. Redirect all
normal entry points to artifact-first native publication, quarantine the local
publisher as an explicit maintenance path, and broaden the architecture ratchet
before deleting compatibility branches.

### M8 — Centralize digest mechanics by exact grammar

The safe rule is to consolidate only byte-identical preimage grammars.

**Strict UTF-8 canonical JSON.** Keep
`src/fisheye/shared/zarr/manifest_digest.py:18-33`. Byte-identical local copies
include:

- `src/fisheye/shared/zarr_payload_receipt.py:37-48`;
- `src/fisheye/cohorts/spec.py:330-343`;
- `src/fisheye/analysis/track_kinematics.py:4466-4478`;
- `src/fisheye/detection/detect_traditional.py:75-86`;
- `src/fisheye/refinement/refine_online_detect.py:292-304`; and
- `src/fisheye/shared/stimulus_coordinate_contract.py:424-434`.

Replace those implementations with imports; no schema bump is required if
fixed vectors prove byte identity.

Keep exact-type JSON validation separate. The nearly identical implementations
at `src/fisheye/analysis/bout_classification_schema.py:418-498`,
`src/fisheye/analysis/tail_posture_view_schema.py:300-380`, and
`src/fisheye/shared/eye_angle_schema.py:1048-1128` reject tuples, NumPy scalar
subclasses, and non-string keys. Extract a stricter shared helper rather than
weakening them to the permissive manifest serializer.

**Array digests.** Preserve two different algorithms:

- raw C-contiguous decoded bytes, currently in
  `src/fisheye/shared/zarr/benchmark_runtime.py:20-23`; and
- typed-array v1, whose canonical header plus NUL plus C-order bytes is in
  `src/fisheye/shared/coordinate_frame_record.py:600-725`.

Move both to a neutral content-digest module with in-memory and bounded
streaming forms. Do not merge them: raw hashes rely on surrounding dtype/shape
declarations.

**Trees.** Atomic-publisher, detection-artifact, benchmark, payload-receipt, and
stat-only tree hashes use incompatible preimages. Build one shared scanner, but
retain explicit versioned adapters. Never hide the differences behind an
ambiguous `tree_hash(mode=...)`.

Add independent frozen vectors before deleting helpers. Existing tests often
compute expected results with the same implementation under test.

Immediate private-helper deletions found by reference search:

- `src/fisheye/group_statistics/goodcopbadcop.py:716` (`_sha256_file`);
- `src/fisheye/analysis/track_kinematics.py:4814`
  (`_motion_group_attrs_sha256`); and
- the unused `_hash_parameters` / `_build_keypoint_signature` chain at
  `src/fisheye/utils/run_keypoints_batch.py:346-402`.

### M9 — Route bespoke lifecycle code through shared primitives

**Custom subject-mask batch publisher.** Replace
`src/fisheye/utils/run_subject_mask_batch_pipeline.py:902-954,1067-1195` with
the shared atomic publisher. The custom path can delete an occupied target,
uses a PID-only temporary name, performs unchecked `copytree`, publishes subject
and refined runs sequentially, and writes completion afterward. Remove normal
`overwrite=True`; immutable publication uses a new name or exact idempotent
reuse.

**Selector engines.** Migrate these duplicated implementations to
`activate_selector_eligible_run`:

- chaser distance:
  `src/fisheye/analysis/chaser_distance_coordinate_publication.py:132-294,2275-2464`;
- subject shape:
  `src/fisheye/shared/subject_shape_coordinate_publication.py:4021-4737`; and
- track, after generalizing the survivor for ordered multi-parent mutations:
  `src/fisheye/analysis/track_kinematics.py:2100-3075`.

Keep owner UUIDs, generation fencing, fresh-path resolution, guarded rollback,
and eligibility as the final write.

**Tombstones.** Ten diagnostic candidate functions and several production
writers independently implement archive lock, fresh path resolution,
ownership/binding checks, `mark_run_failed`, tombstone schema, consolidation,
and reopen validation. Extract one `tombstone_execution_candidate(spec)` and one
owner-checked `transition_owned_run_to_failed` rather than deleting tombstone
behavior.

**Standalone directory publishers.** At least eight training paths repeat local
hash, `copytree`, hidden-target hash, rename, and cleanup. Extract one checked
directory publication kernel; retain family-specific semantic validators.

**Zarr openers/resolvers.** Standardize on
`src/fisheye/shared/zarr_io.py:14-52`. Repoint and delete
`src/fisheye/shared/zarr_helpers.py:413-425`, then retire Zarr-2 `TypeError`
fallback ladders because the dependency contract is Zarr 3-only. Repair the
typed resolver's missing eligibility check at
`src/fisheye/shared/run_resolution.py:265-291`, then route other resolvers
through it. Lexical fallback remains only in explicitly named historical
adapters.

### M10 — Reduce keypoint/crop identity duplication

**Rename digest-only “signed” records.** The following names describe unkeyed
SHA-256 bindings, not signatures:

- `src/fisheye/shared/zarr/crop_consumer.py:38-43`;
- `src/fisheye/shared/hybrid_crop_provider.py:69-86,197-289`;
- `src/fisheye/shared/acquisition_crop_identity.py:28-84,193-209`; and
- `src/fisheye/shared/keypoint_terminal_pixel_evidence.py:16-17`.

Use `content_bound`, `provider_manifest`, or `row_fingerprint` terminology. For
new schemas, remove duplicate `crop_signature` plus constant `crop_revision=1`
wrappers after readers use one crop/provider manifest fingerprint.

**Bound ancestry.** `src/fisheye/shared/zarr/refined_keypoint_manifest.py:346-497`
stores and copies the complete `ancestry_snapshot_ids` list for every successor,
giving O(history-depth) records. No production consumer was found outside its
validator. Retain `lineage_id`, current snapshot, and immediate parent artifact
reference; resolve deeper history by following parents with cycle detection.

**Stop semantic double storage.** Raw, quality, and refined publishers write
semantic documents in ordinary attrs and then embed them again in immutable
manifests:

- `src/fisheye/shared/zarr/keypoint_publication.py:491-566`;
- `src/fisheye/shared/zarr/keypoint_quality_publication.py:259-341`; and
- `src/fisheye/shared/zarr/refined_keypoint_publication.py:364-463`.

Move consumers to one manifest accessor. New schemas retain only
lifecycle/discovery attrs and the manifest.

**Clip receipts.** Preserve one terminal worker record, but replace whole-crop
rehashing per clip at
`src/fisheye/utils/write_keypoint_clip_terminal_receipt.py:244-267` and
`src/fisheye/utils/finalize_clipped_keypoint_v2_bundle.py:187-200` with crop
artifact fingerprint, selected row IDs, selection/row digest, output digests,
and model/preprocessing fingerprints.

**Shared publisher.** Raw, quality, and refined keypoint publishers duplicate
physical-unit writes, consolidation, manifest install, validation, and reopen
logic at `keypoint_publication.py:343,470-611`,
`keypoint_quality_publication.py:93,245-394`, and
`refined_keypoint_publication.py:128,345-538`. Extract one immutable-Zarr
snapshot publisher with family-specific schema hooks.

The older `src/fisheye/shared/keypoint_coordinate_publication.py` is a
3,430-line overlapping attr-proof and selector system. Migrate its production
readers/writers to strict manifests and shared activation before retiring its
proof/activation grammar; preserve numerical coordinate projection helpers.

**[W] Dirty-path warning.** New direct-strict receipt support at
`src/fisheye/utils/write_keypoint_clip_terminal_receipt.py:170-214` is tested
only at construction. The committed finalizer still assumes legacy crop attrs
and a physical `source_crop_row_ids` array at
`src/fisheye/utils/finalize_clipped_keypoint_v2_bundle.py:162-196`. Add an
end-to-end strict clip-to-receipt-to-bundle test before treating the new profile
as complete or deleting the legacy path.

---

## Machinery that is not redundant

Do not remove these while simplifying the envelopes around them:

- atomic same-parent rename, archive locking, owner fencing, generation checks,
  fresh-path resolution, guarded rollback, and eligibility as the final write;
- immutable-publication root consolidation after payload, provenance, selector,
  and eligibility state are final;
- one local scientific validation, exact copy-equivalence proof, and final fresh
  pre-selector validation;
- subject-mask row-unit hashing, gap-free worker coverage, streaming source-byte
  revalidation, crop/source row joins, and final-layout ownership checks;
- clip-terminal keypoint receipts that prove each expected distributed work unit
  ended in success or failure;
- row-level fingerprints used for partial reuse and stale-edit detection;
- stage-specific semantic validators and the exhaustive audit-reader path; and
- legacy digest interpreters until archive/schema inventories permit retirement.

---

## Subtraction implementation order

1. Land independent digest vectors and broaden architecture/reference ratchets.
2. Apply S1-S3 and stop the duplicate writes in S5-S6 while retaining legacy
   readers.
3. Decide whether S4 is a supported external API; delete it if not.
4. Redirect all normal detection entry points to artifact-first publication.
5. Introduce new manifest/receipt versions and dual-write compact identity plus
   validation evidence.
6. Migrate consumers away from track commit, detection candidate/shadow
   receipts, subject-mask nested evidence, and keypoint semantic attrs.
7. Replace `cluster_output_staging` authority with compact publication evidence.
8. Split publisher callbacks into scoped validations and enforce read/hash-count
   tests before removing repeated scans.
9. Route custom publishers, selectors, tombstones, openers, and resolvers through
   shared kernels.
10. Run archive and selector censuses, then retire fail-open compatibility and
    old schema writers.

### Minimum focused regression suites

Per repository policy, run real-Zarr suites outside the Codex sandbox:

```bash
# Track envelopes and scan reduction
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_zarr_payload_receipt.py \
  tests/unit/fisheye/test_track_motion_publication.py \
  tests/unit/fisheye/test_track_kinematics_coordinate_contract.py \
  tests/unit/fisheye/test_track_kinematics_materializer.py \
  tests/unit/fisheye/test_atomic_run_publisher.py -q

# Subject-mask evidence consolidation
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_subject_mask_coordinate_validation_receipt.py \
  tests/unit/fisheye/test_coordinate_successor_publication.py \
  tests/unit/fisheye/test_subject_mask_coordinate_receipt_loader.py \
  tests/unit/fisheye/test_refined_subject_mask_coordinate_receipt_loader.py -q

# Detection receipt/report separation
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_native_canonical_detection_candidate.py \
  tests/unit/fisheye/test_native_canonical_detection_publication.py \
  tests/unit/fisheye/test_refined_detection_shadow.py \
  tests/unit/fisheye/test_detection_snapshot_publication.py \
  tests/unit/fisheye/test_detection_publication_architecture.py -q

# Keypoint/crop reduction
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_keypoint_successor.py \
  tests/unit/fisheye/test_clipped_keypoint_finalization.py \
  tests/unit/fisheye/test_clipped_keypoint_v2_finalization_workflow.py \
  tests/unit/fisheye/test_refined_keypoint_publication.py -q
```

Add count-based tests for reads, tree scans, and hashes. Prefer those over
wall-clock thresholds. Also add frozen legacy fixtures proving old manifests and
receipts remain readable while no new writer emits them.

---

## Validation performed

**[V]** Read-only checks run during the review:

```text
git diff --check
# pass

scripts/py scripts/check_zarr_open_group_modes.py --no-update-on-shrink
# fail: implicit metadata mode at
# src/fisheye/utils/materialize_clipped_keypoint_direct_hybrid_terminal.py:243
```

Post-integration document checks found:

- no trailing-whitespace diagnostics in the new file;
- all 98 cited repository paths exist; and
- all 136 parsed `path:start-line` references begin within the cited file.

No pytest suite was run because this change creates only a diagnostic document.
No files other than this document were intentionally created or modified by the
review.
