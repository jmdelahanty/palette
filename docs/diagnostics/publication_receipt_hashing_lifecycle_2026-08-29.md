# Publication receipt and hashing lifecycle — 2026-08-29

Status: **design decision and optimization contract**. This governs new work;
existing publication families may require a compatibility migration before
their redundant payload scans can be removed.

Legacy behavior was observed against Palette commit `bf058521`
(`subject-shape-unbound-receipt-20260827-bf058521`). The receipt-v2
implementation checkpoint is `e35714b0`.

## Decision

When a trusted receipt covers the final immutable payload, normal publication
must not decode or hash that payload again.

Payload identity is established while data is written or transformed.
Publication verifies the resulting receipt chain, metadata, authority,
ownership, completion, and atomic placement. A complete decoded rehash remains
an explicit audit or recovery operation, not a routine planning, publication,
or consumption step.

This is not a reduction in validation strength. It moves the integrity proof to
the point where the data is already in memory and makes the proof reusable at
later boundaries.

## Required lifecycle

### 1. Planning

Planning is read-only and cheap. It resolves supported producer profiles and
already-published input receipts. It may validate metadata and receipt seals,
but it does not read large payload arrays, create output receipts, or predict
digests for data that has not been written.

### 2. Writing and transformation

The process that creates or transforms payload bytes records their identity as
part of the same attempt. Each write unit must bind at least:

- the publication attempt and owned row/chunk interval;
- array path, shape, dtype, physical chunking, and codec declaration;
- row identity and relevant scientific/coordinate authority;
- the exact transformation and implementation/profile version; and
- a digest of the final bytes or logical values governed by that receipt.

Parallel writers may issue receipts only for whole, non-overlapping physical
chunks they own. Logical row slices that share a physical Zarr chunk are not
independent write units.

### 3. Finalization

Finalization proves that the expected write-unit receipts are complete,
non-overlapping, ordered, and bound to one attempt. It deterministically seals
them into a publication receipt.

A flat SHA-256 of a complete array cannot be reconstructed from independent
SHA-256 chunk digests. Parallel output therefore needs a versioned composable
identity, such as a chunk/Merkle manifest, or a single ordered writer that
computes the required flat digest while writing. The receipt schema must state
which identity it uses; consumers must never silently treat one as the other.

### 4. Atomic publication

Routine publication validates:

- receipt schema and seal;
- completeness and exact write-unit coverage;
- attempt, producer, source-authority, and row-domain bindings;
- array topology and metadata declarations;
- expected publication path and ownership;
- completion and selector policy; and
- the final consolidated metadata generation.

An atomic rename on the same filesystem does not change payload identity and
therefore does not require a payload rehash. A copy or cross-storage transfer
must produce a transfer receipt bound to the source publication receipt. If the
transport cannot prove the transfer, a destination-side verification scan is
required once.

### 5. Consumption

Normal consumers validate the sealed publication receipt and the metadata
generation required by the publication contract. They must not routinely
recompute full payload digests already proven by that receipt.

### 6. Deep audit

Full decoded equality checks, re-derivation, and payload rehashes remain
available as explicit operations. They are required when:

- the writer did not issue a compatible receipt;
- any payload or covered metadata changed after receipt creation;
- publication performed an unreceipted transformation;
- a transfer lacks sufficient integrity evidence;
- the receipt is absent, incomplete, invalid, or uses an unsupported profile;
- recovery is being attempted; or
- an operator explicitly requests a deep audit.

These conditions fail closed. They must not trigger an unreported expensive
fallback during ordinary planning or publication.

## Baseline defect demonstrated by the subject-shape canary

The commit-`bf058521` subject-shape writer does not emit reusable evidence for
its final parallel output. Finalization consequently rereads completed arrays in
`build_subject_shape_unbound_payload_scan_receipt` to manufacture flat payload
hashes. Other admission paths have also reloaded payloads to reproduce claims
already sealed by upstream receipts.

That scan is compatibility work caused by a missing write-time receipt; it is
not subject-shape computation and is not the intended steady-state publication
design.

The durable optimization is to make the already-required storage transform
issue the write-unit receipt while it has source and destination values in
memory, then teach publication and consumers to accept that versioned
composable identity.

## Subject-shape v2 implementation checkpoint

Status: **implemented and locally validated in
`agent/palette/subject-shape-plan-receipts-20260829` through `6d3d3c0d`; pushed
to draft PR #73. All 23 required CI checks passed at that runtime commit and
the selector-ineligible receipt and authority-proof performance canaries
completed successfully. The canaries also exposed a separate pre-existing
nondeterministic ellipse-fit edge case, so they remain ineligible and the
branch is not merged or production-active**.

The projected access-aware subject-shape path now implements the first durable
receipt profile:

`verified_staged_transfer_plus_binding_append_receipt_v2`.

The proof chain is:

1. The parallel compute writer closes its scratch tree with a sealed,
   metadata-only deferred-transform receipt. It does not invent or predict any
   payload digest. That receipt can be consumed only by the named access-aware
   storage profile under exclusive scratch ownership.
2. Access-aware storage conversion writes complete physical outer units,
   immediately reads each unit back, and emits the closed decoded copy report
   and flat per-array content digests from that same traversal. Exact decoded
   source-to-destination equality is checked before the receipt closes.
3. Only after every copy/readback succeeds does conversion consume the deferred
   receipt and stamp the ordinary sealed unbound manifest. A missing, changed,
   leftover, or profile-mismatched deferred receipt fails closed.
4. The shared atomic publisher passes its already-verified physical-copy
   receipt into the existing post-rename callback. This is one shared
   publication interface; no subject-shape-specific transfer adapter was
   added.
5. Final-path binding verifies that the numeric projection is already in
   source-camera coordinates and therefore does not rewrite any staged array.
6. Binding creates and reads back only the six appended arrays:
   `instance_key`, `source_crop_row_ids`,
   `source_acquisition_frame_index`, `component_centroid_xy`,
   `component_centroid_valid`, and `body_frame/axis_valid`.
7. The staged and appended decoded receipts must partition the exact live
   array inventory. Missing, overlapping, or unexpected arrays fail closed.
8. The final target receives one physical-payload hash and immutable-metadata
   receipt. The scientific validation receipt binds that final integrity root,
   the coordinate manifest, the transfer evidence, and the decoded receipt
   composition.

The old `single_locked_bound_payload_receipt_v1` grammar remains readable and
continues to be emitted for paths that genuinely transform numeric geometry at
final binding or do not supply the v2 transfer evidence. Its exported constant
retains its historical meaning; v2 has a distinct explicit constant and
profile marker.

This removes both complete decoded rescans from the maintained projected
access-aware path: the post-compute unbound scan and the post-binding scan. It
deliberately does **not** remove the final physical-file hash yet: that scan
covers the low-single-digit-gigabyte compressed tree, not the hundreds of
gigabytes delivered by decoded array traversal. Reusing physical leaves across
transport is a smaller follow-up and requires the atomic transport to expose
compatible per-file leaves, not merely an aggregate copy digest.

Local evidence so far:

- 62 complete affected atomic-publisher and subject-shape tests passed in
  425.92 seconds;
- the final instrumented access-aware regression passed in 189.94 seconds;
- that regression compares every v2 declared final array digest with the live
  final array, guarding against reuse of a staged digest after a binding write;
  and
- that regression makes both
  `build_subject_shape_unbound_payload_scan_receipt` and
  `_scan_subject_shape_bound_payload` raise if the v2 path attempts either
  complete decoded rescan.

### Completed legacy baseline

LSF job `153770521` ran the immutable commit-`bf058521` implementation against
Cam2010094 and completed successfully on 2026-08-29 at 22:17:31 UTC. The run
`subject_shape_fused_projection_canary_20260829_bf058521_cam2010094` is
complete, selector-ineligible, bound under `bound_canonical_v2`, and sealed
with the historical `single_locked_bound_payload_receipt_v1` profile. Registry
writes were disabled.

| Phase | Wall time | Delivered read characters |
|---|---:|---:|
| plan | 1,403.552 s (23m24s) | 37.31 GB |
| scientific compute, including the old post-compute receipt scan | 6,764.819 s (1h52m45s) | 396.67 GB |
| access-aware storage conversion | 361.063 s (6m01s) | 2.78 GB |
| atomic publication | 9,826.970 s (2h43m47s) | 826.43 GB |
| cleanup | 22.871 s | negligible |

Within atomic publication, `post_rename_binding` alone took 9,748.514 seconds
(2h42m29s) and delivered 818.42 GB of reads. The whole job took 18,390.019
seconds (5h06m30s), delivered 1,460,813,022,742 read characters, and averaged
2.507 effective CPU cores across 32 allocated slots: 7.835% slot efficiency.
The OS reported only 6,487,134,208 physical storage-read bytes, so delivered
read characters primarily measure repeated decode/cache traversal rather than
network or physical-disk traffic.

A metadata-file census of the baseline target found 106 arrays totaling 10.266
GiB of logical values. Its final physical-integrity receipt covers
1,439,980,758 bytes (about 1.34 GiB). Therefore the 1.461 TB of delivered read
characters was about 133 times the complete logical payload and about 1,014
times the final compressed tree. Delivered read characters include page-cache
traffic and are not physical storage bytes, but the ratios still expose
repeated sharded decode traversal rather than one sequential verification pass.

### Completed receipt-v2 canary

LSF job `153770860` ran commit `e35714b0` against the same Cam2010094 inputs,
on the same host and 32-slot resource request. It started immediately after the
baseline at 22:17:31 UTC, completed with exit code zero at 23:03:58 UTC, and did
not overlap the baseline. The run
`subject_shape_receipt_v2_canary_20260829_e35714b0_cam2010094` remains
selector-ineligible, registry writes were disabled, and the consolidated root
contains the completed `bound_canonical_v2` publication.

| Phase | Legacy `bf058521` | Receipt v2 `e35714b0` | Speedup |
|---|---:|---:|---:|
| plan | 1,403.552 s | 23.080 s | 60.81x |
| scientific-compute envelope | 6,764.819 s | 995.189 s | 6.80x |
| access-aware storage conversion | 361.063 s | 348.977 s | 1.03x |
| atomic publication | 9,826.970 s | 1,299.618 s | 7.56x |
| post-rename binding | 9,748.514 s | 1,222.107 s | 7.98x |
| total scheduler-observed job | 18,390.019 s | 2,786.824 s | 6.60x |

Total wall time fell from 5h06m30s to 46m27s, an 84.85% reduction. Delivered
read characters fell from 1,460,813,022,742 to 475,337,630,238, a 67.46%
reduction. Average effective CPU use increased from 2.507 to 9.950 cores and
32-slot efficiency rose from 7.835% to 31.095%, approximately 3.97 times the
baseline efficiency.

The receipt chain closed as designed:

- the final profile is
  `verified_staged_transfer_plus_binding_append_receipt_v2`;
- all 100 storage-transform array digests equal the corresponding 100 carried
  final digests;
- the six declared binding arrays are the only appended paths;
- the final receipt records
  `verified_staged_transfer_plus_final_binding_readback_v2` as its digest
  source;
- the physical-copy receipt covers 1,365,323,664 bytes in 1,168 files and was
  verified by an `rsync --checksum` dry run; and
- the published consolidated metadata records completion, selector
  ineligibility, canonical binding, and the v2 profile.

There is no hidden legacy decoded-payload scan. The remaining 1,222.107-second
binding interval is dominated by `authority_stamping`: 912.950 seconds and
43.008 GB of delivered read characters. The final `physical_payload_hash` took
only 2.912 seconds and delivered 1.441 GB. The authority interval wrote only
90.876 MB, which points to large-metadata read/modify/write amplification, not a
second numeric transformation. Admission revalidation (113.308 seconds),
coordinate-source binding/projection (111.018 seconds), and identity-array
append (69.914 seconds) are the other material subphases. Reducing repeated
authority-metadata serialization was therefore the next publication
optimization; the follow-up below removes the repeated source rebuilds without
removing receipt checks.

### Completed authority-proof follow-up

The first measured follow-up, commit `c94426a7` and LSF job `153771730`, made
canonical descriptor attributes use one fail-closed whole-map Zarr attribute
replacement and instrumented each authority subphase. It completed in
2,788.240 seconds, statistically neutral against the receipt-v2 canary. That
negative result was useful: `authority_coordinate_descriptors` still took
680.203 seconds while writing only 143,899 characters, proving that the
dominant work occurred before attribute mutation.

The root cause was repeated bundle-source verification. Each descriptor called
`require_bound_subject_shape_bundle_source()`, whose bound source reopened and
rebuilt the full sealed subject-mask bundle authority on every invocation.
Commit `6d3d3c0d` routes that existing `assert_verified()` implementation
through the shared operation-scoped persisted-proof interface. The first call
per operation runs the full verifier, all intermediate calls reuse the sealed
result, and scope close runs one fresh full verifier for time-of-check/time-of-
use protection. No source digest, receipt, authority, or selector rule was
removed.

LSF job `153771776` ran `6d3d3c0d` against the same Cam2010094 inputs, on the
same `h07u20` host and 32-slot request as `153771730`. It completed with exit
code zero in 1,837.461 seconds (30m37s): 15m51s faster than the whole-map
control and a 34.1% end-to-end reduction.

| Phase | `c94426a7` | `6d3d3c0d` | Speedup |
|---|---:|---:|---:|
| scientific compute | 983.161 s | 977.537 s | matched |
| access-aware storage conversion | 347.284 s | 350.810 s | matched |
| atomic publication | 1,310.150 s | 359.120 s | 3.65x |
| authority stamping | 913.267 s | 117.417 s | 7.78x |
| coordinate descriptors | 680.203 s | 0.369 s | 1,845x |
| total job | 2,788.240 s | 1,837.461 s | 1.52x |

Authority-stamping delivered reads fell from 45.319 GB to 3.877 GB. Whole-job
delivered reads fell from 480.002 GB to 431.796 GB, and 32-slot CPU efficiency
rose from 30.583% to 44.974%. The optimized output is complete,
`bound_canonical_v2`, selector-ineligible, registry-inactive, and carries the
same source-binding, row-identity, placement-array, numeric-projection, bundle,
and storage-profile identities. Run-local record digests differ because sealed
record references include the new output run path.

The remaining authority cost is now visible rather than obscured:
`authority_body_frame` takes 103.234 of the 117.417 seconds. The physical
payload hash remains 2.937 seconds. Any later body-frame optimization must
preserve its complete scientific and coordinate validation.

### Scientific parity finding

The receipt-v2 canary has the same 106-array topology as the baseline. Ninety-
eight array digests are byte-identical. Eight differ, all downstream of
`cv2.fitEllipse`:

| Component/output | Differing rows or elements |
|---|---:|
| left-eye ellipse parameters | 31 rows |
| left-eye ellipse success | 29 elements |
| left-eye body-axis angle | 30 elements |
| right-eye ellipse parameters | 87 rows |
| right-eye ellipse success | 75 elements |
| right-eye body-axis angle | 85 elements |
| swim-bladder ellipse parameters | 1,651 rows |
| swim-bladder ellipse success | 1,249 elements |

The source bindings, worker count, chunk size, native-thread limit, and mask
publication digests are identical. A bounded replay against the immutable
source masks found that every affected mask contains only 4--12 foreground
pixels. More decisively, two consecutive calls to the current ellipse fitter on
the same in-memory mask agreed for only 19/31 left-eye rows, 53/87 right-eye
rows, and 754/1,651 swim-bladder rows. Replay variably matched the baseline,
the optimized run, or neither. This is numerical instability in OpenCV's
ellipse fit on degenerate tiny contours, not a transfer-receipt or storage-copy
failure.

Do not choose one canary's unstable values as canonical. The scientific
hardening follow-up is to define a deterministic degenerate-mask admission
predicate before `cv2.fitEllipse`, emit an explicit component failure reason,
and test repeated-run identity. This matches the existing policy that a
component-level geometry failure is recorded without invalidating an otherwise
valid publication. Until that policy lands, this canary remains
selector-ineligible even though the receipt-v2 performance and composition
checks passed.

Follow-up status (2026-08-30): the implementation and evidence are documented
in `docs/diagnostics/subject_mask_component_area_support_2026-08-30.md`. The
candidate combines an exact model/training-bound refinement floor with a
13-foreground-pixel defensive ellipse guard, preserves raw inference evidence,
and records component failure without invalidating the publication. The canary
must remain selector-ineligible until that change completes required CI and a
new successor is explicitly validated; this note does not activate it.

## Eye-angle admission-receipt reuse (2026-08-31)

The eye-angle materializer had a narrower repeat-read defect: a reviewed
plan-only invocation could compute and report the exact staged-input integrity
receipt, but a later apply invocation always rebuilt the plan and recomputed
that receipt. The apply therefore repeated the expensive source scan even when
the operator was deliberately executing the already-reviewed plan.

The implementation candidate adds one sealed operation envelope,
`palette.eye_angle_materialization_admission_receipt` v1, around the complete
typed `EyeAngleMaterializationPlan`. Its payload has a separate
`palette.eye_angle_materialization_plan` v2 identity so the historical
materialization-v1 report plan remains byte-for-byte schema-compatible. The
receipt payload includes the full selected
physical-file manifest, source and authority contracts, the existing staged
scientific-input receipt, all scientific/storage parameters, resolved paths,
selector snapshots, and the exact clean Palette Git commit. Plan-only mode can
write the envelope create-only; apply mode can consume it through the same
materializer interface.

Receipt-backed apply is fail-closed:

- envelope, payload, nested authority, and inventory digests are validated;
- the current process must run the exact clean commit that created the plan;
- every repeated apply argument must agree with the sealed plan;
- live lifecycle, authority, selector, metadata, and selected-file closure are
  revalidated before scratch creation;
- canonical subject-shape sources must also pass direct/consolidated metadata
  equivalence at receipt admission (candidate and keypoint paths retain their
  existing profile-specific metadata gates);
- invalid or stale receipts never fall back to live replanning; and
- the receipt-bound scratch directory remains the atomic single-consumer claim.

The self-contained `apply_eye_angle_materialization_plan` API is the shared
consumer. The eye-gaze prerequisite cohort now writes the receipt after its
subject-shape candidate exists, applies that exact receipt, and binds both the
receipt-file digest and payload digest into its version-2 cohort receipt.

The initial 2026-08-31 optimization removes the *second plan construction* when a prior
approval/dry-run is part of the workflow. It does not claim that metadata reads
are zero, and it does not remove the exact staged-payload verification before
the scientific writer or the closing source audit around publication. A
same-size source mutation with a restored modification time is consequently
caught before computation even though the initial reuse gate is intentionally
metadata/inventory based.

The physical path/size/mtime manifest is only a cheap freshness hint; it is not
described as a standalone content receipt. Scientific content remains bound by
the nested logical-input receipt and its exact post-copy verification. The
scratch claim prevents concurrent consumers, while the immutable target path
prevents successful replay. The receipt itself does not yet carry a durable
claimed/running/committed lifecycle token; add one only if receipt single-use
independent of target identity becomes a workflow requirement.

The 2026-08-31 camera-2010094 canary at commit `1b6d14c0` was a one-shot apply
and did not use this new cross-invocation path. It remains the baseline: its
reported plan phase was about 18m47s, followed by about 18m45s of staged-input
validation.

The receipt-backed camera-2010094 canary at merged commit `68d4edbd` completed
successfully on 2026-09-01. It consumed the exact sealed receipt with payload
digest `69bcd22355e6e6c4349b9ac0f36c943ef25b8200426e5c770a5dc0a2be3c34d8`,
published a valid selector-visible run, and updated the registry. Its 64m38s
wall time also isolated the remaining redundancy: about 18m42s before staging,
about 18m47s after the 0.99s physical stage copy, and 18m33.755s in
`post_rename_binding`. Scientific computation took 448.863s and the access-aware
sharding transform took 5.971s. The three approximately equal serial intervals
are decoded source audits, not planning, copying, sharding, or eye-angle
computation. They are the production calls retired by the follow-up below.

### Worker-consumption receipt and old-path retirement (2026-09-01)

The follow-up implementation removes the remaining production ambiguity and
the repeated decoded scans:

- `materialize_eye_angles(..., apply=True)` now requires an exact admission
  receipt at the shared Python API boundary, before planning or scratch
  creation. The CLI inherits that gate instead of maintaining a separate
  policy check.
- The storage-candidate executor now plans, seals, and consumes the same
  admission receipt. No maintained source caller can execute a fresh-plan
  one-shot apply.
- Receipt construction reads each worker chunk once. During that owned-snapshot
  pass it emits the per-chunk receipts and streams full-array canonical digests
  back to the sealed subject-shape and keypoint authorities. The immediate
  second full decoded pass and the later planner-only frame-index rescan are
  removed.
- Staging, startup, and closing checks use the same receipt-bound resolver in
  metadata/authority/inventory mode. They validate lifecycle, closed topology,
  paths, shapes, dtypes, nested authorities, metadata generations, and receipt
  seals without decoding the scientific payload again.
- Every compute worker hashes the owned C-order snapshot it actually consumes
  before scientific computation. Publication requires a sealed attestation for
  the exact complete ordered receipt-chunk set; missing, duplicate, reordered,
  or foreign receipts fail closed.
- A full decoded source scrub remains available only through the explicitly
  named diagnostic `audit_eye_angle_source_revision_full_payload`. The layout
  benchmark invokes that diagnostic deliberately; normal production does not.

The removal boundary is deliberate: diagnostic capability remains available,
but routine callers cannot opt back into it with a mode flag. The old
fresh-plan one-shot executor call and the three routine full-scan invocations
are absent from maintained execution paths; unreceipted apply is rejected at
the shared Python API before planning or scratch creation.

This is not a weaker copy-integrity policy. A mutation before a worker reads its
chunk differs from the receipt and stops the workload. A mutation after that
worker has copied, hashed, and computed from its immutable in-memory snapshot
does not invalidate the already-proven scientific result. Partial output never
becomes complete or selector-visible because the complete worker attestation is
required by local validation and every publication gate.

The acceptance boundary is guarded by tests that prove one receipt-builder
snapshot read per chunk, exact streaming-digest parity with the canonical array
digest grammar, restored-mtime staged corruption rejection in the worker,
complete worker-set enforcement, real materializer publication with persisted
attestation, and rejection of unreceipted apply before planning.

## Compatibility and migration

Existing consumers may require a historical flat content digest. During a
bounded migration, a publisher may carry both:

- the existing flat digest, computed by the one unavoidable compatibility
  scan; and
- the new composable write-time receipt used by updated publication and
  consumers.

Dual identity is transitional. Each publication profile adopting it needs a
written removal condition for the flat rescan and a real
writer -> finalizer -> publisher -> unpatched consumer boundary test.

Receipt profiles remain stage-specific. Subject masks, subject shape, and
keypoints can share low-level digest, chunk-ownership, and atomic-publication
machinery, but their receipts must declare their distinct scientific and
authority claims.

## Instrumentation requirements

Publication timing must separate:

- transformation and payload writing;
- write-unit receipt creation;
- receipt aggregation and sealing;
- final decoded scan, if compatibility still requires one;
- physical transfer or rename;
- metadata and authority stamping;
- metadata consolidation; and
- selector/registry finalization.

Record bytes read and written, CPU time, wall time, worker count, requested and
effective chunk ownership, receipt profile, and whether a deep scan ran. A
multi-hour scan must never be reported merely as "planning" or "publication."

## Acceptance checklist

- [ ] The writer emits sealed receipts for the exact final write units.
- [ ] Parallel write units align to whole, non-overlapping physical chunks.
- [ ] Finalization detects missing, duplicate, overlapping, reordered, or
      foreign-attempt receipts without opening payload arrays.
- [ ] Routine publication performs zero decoded payload reads after a valid
      final receipt exists.
- [ ] Same-filesystem atomic rename performs no payload rehash.
- [ ] Transfer publication either validates a bound transfer receipt or runs
      exactly one explicitly reported destination verification scan.
- [ ] Normal consumers validate receipts and published metadata generation
      without reproducing payload hashes.
- [ ] Explicit deep audit recomputes the payload and detects tampering.
- [ ] Receipt or metadata tampering fails closed.
- [ ] Counting-store and phase telemetry prove that redundant reads are gone.
- [ ] A real writer -> finalizer -> publisher -> strict unpatched consumer test
      guards each supported receipt profile.
