# Subject-shape planning admission optimization — 2026-08-29

Status: **implemented through `6d3d3c0d`, locally validated, pushed to draft
PR #73, and exercised by completed selector-ineligible production-scale
canaries. All 23 required CI checks passed at the runtime commit. A separate
pre-existing ellipse-fit nondeterminism finding still requires scientific
hardening; not merged or production-active**.

Worktree branch: `agent/palette/subject-shape-plan-receipts-20260829`

Legacy baseline canary commit: `bf058521`

Optimized canary commit: `e35714b0`

Whole-map authority-stamping canary commit: `c94426a7`

Operation-scoped authority-proof canary commit: `6d3d3c0d`

This optimization follows the shared receipt lifecycle decision in
[`publication_receipt_hashing_lifecycle_2026-08-29.md`](publication_receipt_hashing_lifecycle_2026-08-29.md).
It first improved input admission and planning, then implemented the related
subject-shape receipt-v2 publication path. The publication implementation is
described in the linked lifecycle document; it removes the post-compute and
post-binding complete decoded rescans from the maintained projected
access-aware path.

## Problem

A read-only subject-shape plan for the 2,745,488-row Cam2010094 recording was
performing finalization-grade work while admitting already-published inputs.
The observed amplification included:

- rereading bounded samples from subject-mask core, quality, and presentation
  cache payloads even though the bundle sealed their publication receipts;
- reopening the same subject-mask coordinate authority during assignment
  rebinding validation;
- parsing the large root consolidated metadata snapshot repeatedly; and
- loading and re-deriving the complete keypoint coordinate surfaces merely to
  verify a sealed rebinding dependency.

The last item was the remaining dominant defect. Planning needed proof that the
named coordinate successor was still the admitted immutable authority; it did
not need its millions of scientific coordinate values.

## Implementation

### Receipt-backed subject-mask member admission

Normal bundle admission now validates the member manifests, lifecycle,
topology, metadata declarations, sealed digests, and bundle bindings without
recomputing bounded payload samples. The existing public deep validators remain
unchanged and continue to read payload values when explicitly invoked.

The bundle reader also shares one consolidated root metadata snapshot across
the core, quality, cache, and bundle metadata checks.

### Reuse of an already-bound mask authority

Subject-shape planning passes its sealed
`BoundRecordingSubjectMaskCoordinateAuthority` into assignment-rebinding
validation. The rebinding loader revalidates that sealed object and its exact
archive identity instead of independently loading the complete bundle a second
time.

### Shared keypoint successor admission

`load_keypoint_coordinate_successor_admission()` now owns the common profile
grammar:

- exact run path and manifest validation;
- coordinate-successor authority and source-authority binding;
- completion, selector-ineligible candidate, and parent policy checks;
- active keypoint bundle authority agreement; and
- direct/consolidated metadata declaration validation.

It returns a sealed metadata-only admission and does not open coordinate
payload surfaces.

`load_keypoint_coordinate_successor_source()` invokes that same admission first
and then loads and validates the full coordinate surfaces. Therefore planning
and scientific consumption share one authority grammar; the optimization does
not introduce a second resolver or a weaker mode. Rebinding publication and
equivalence inspection still use the full loader. Only later dependency
admission uses the metadata-only result.

### Operation-scoped subject-mask authority proof

The first authority-stamping pass replaced repeated per-field Zarr attribute
writes with one fail-closed whole-map replacement and added nested authority
phase telemetry. Its matched `c94426a7` canary was intentionally retained even
though total runtime was neutral: the telemetry proved that attribute writes
were not the dominant cost. `authority_coordinate_descriptors` spent 680.203
seconds before writing only about 144 KB.

The actual amplification was in
`BoundSubjectShapeBundleSource.assert_verified()`. Every one of 32 coordinate
descriptors, plus several other authority helpers, independently reopened and
rebuilt the same subject-mask bundle source. The shared persisted-proof
mechanism now gives that bound source operation-scoped verification:

1. first use performs the complete bundle-source validation and seals its
   source digest in the operation scope;
2. intermediate consumers reuse that exact proof; and
3. scope close performs one fresh complete validation to retain fail-closed
   time-of-check/time-of-use protection.

This is two authority-bundle validations per publication operation, not two
decoded subject-mask payload scans. A real writer-to-publisher regression calls
the bound source three times inside one proof scope and asserts exactly two
loader invocations: initial and closing validation. The closing validation
still fails if the source digest changes. Activation state is deliberately not
part of scientific source identity, preserving the existing rule that an
already-bound immutable source remains valid after a later selector change.

## Measured production-scale result

Read-only command inputs:

- recording: `2026_08_06_19_13_35_cam2010094`;
- bundle:
  `subject_mask_bundle_sleepyfish_2026_08_06_hybrid_pose_subject_masks_20260821_v002_sleepyfish_cam2010094`;
- rebinding:
  `assignment_keypoint_rebinding_sleepyfish_2026_08_06_direct_hybrid_20260826_v002_sleepyfish_cam2010094`;
- storage profile: `subject_shape_access_aware_candidate_v1`; and
- rows: `2,745,488`.

| Measurement | Partial optimization | Current worktree |
|---|---:|---:|
| command wall time | 677.871 s | 20.646 s (`e35714b0`) |
| plan telemetry wall time | not separately retained | retained in the admission plan |
| delivered read characters | more than 26 GB observed from `/proc` | 1,184,091,249 bytes |
| peak RSS | not retained | 917,776 KiB |
| result | valid plan | valid plan with the same source binding |

The initial optimized checkpoint completed in 18.73 seconds, approximately
**36.2 times faster** than the reproduced partial-optimization baseline. The
final instrumented `e35714b0` checkpoint completed in 20.646 seconds and
retained the same bundle, rebinding, row count, source-binding digest,
coordinate-authority digest, and member-manifest digests. Its sealed admission
plan digest is
`3041029ad8294d103ad4f6b7fe4e44def86949a16069244f8a358c0224f833cc`.

The remaining approximately 1.18 GB of delivered metadata reads are visible and
should be investigated separately. They are no longer a reason to block the
four-camera analytics delivery, but root-metadata parsing and declaration-map
construction remain candidates for a later bounded optimization.

## Completed full-workload comparison

The dependency-serialized Cam2010094 jobs used the same host, source bundle,
rebinding, 32 slots, 1,024-row worker blocks, and one native thread per worker.
Both outputs are complete, selector-ineligible, and registry-inactive.

| Measurement | Legacy `bf058521` | Receipt v2 `e35714b0` |
|---|---:|---:|
| total wall time | 18,390.019 s (5h06m30s) | 2,786.824 s (46m27s) |
| plan | 1,403.552 s | 23.080 s |
| scientific-compute envelope | 6,764.819 s | 995.189 s |
| storage conversion | 361.063 s | 348.977 s |
| atomic publication | 9,826.970 s | 1,299.618 s |
| post-rename binding | 9,748.514 s | 1,222.107 s |
| delivered read characters | 1.461 TB | 475.338 GB |
| 32-slot CPU efficiency | 7.835% | 31.095% |

This is a 6.60x total speedup and an 84.85% wall-time reduction. The storage
transform is intentionally almost unchanged: it is the necessary access-aware
rewrite and exact readback. The speedup comes from receipt-backed planning and
removing the two redundant complete decoded scans.

The v2 canary's 100 transform-produced digests exactly equal the 100 carried
digests in the final receipt, and only the six declared authority/binding arrays
were appended. Consolidated metadata contains the completed, canonical-bound,
selector-ineligible v2 publication.

The comparison also exposed an independent scientific issue. Eight of 106
baseline/optimized array digests differ, exclusively for ellipse parameters,
ellipse success, and the two derived eye-axis angles. All affected component
masks have only 4--12 foreground pixels, and repeated calls to
`cv2.fitEllipse` on the same in-memory mask are not stable. The receipt path did
not cause this: the other 98 arrays are byte-identical and the transfer receipt
closes exactly. A deterministic degenerate-mask rejection contract and explicit
failure reason are required before either set of unstable ellipse values can be
treated as canonical.

## Completed authority-proof canary

LSF job `153771776` ran commit `6d3d3c0d` against the same Cam2010094 bundle,
rebinding, host (`h07u20`), 32-slot allocation, worker count, chunking, storage
profile, and registry-disabled selector-ineligible policy as the matched
`c94426a7` authority-stamping control (job `153771730`). Both runs completed
successfully.

| Measurement | Whole-map control `c94426a7` | Proof reuse `6d3d3c0d` | Change |
|---|---:|---:|---:|
| total wall time | 2,788.240 s | 1,837.461 s | 1.52x; -15m51s |
| plan | 23.758 s | 24.281 s | neutral |
| scientific compute | 983.161 s | 977.537 s | neutral |
| storage conversion | 347.284 s | 350.810 s | neutral |
| atomic publication | 1,310.150 s | 359.120 s | 3.65x |
| post-rename binding | 1,231.648 s | 281.297 s | 4.38x |
| authority stamping | 913.267 s | 117.417 s | 7.78x |
| coordinate descriptors | 680.203 s | 0.369 s | 1,845x |
| delivered read characters | 480.002 GB | 431.796 GB | -48.206 GB |
| average effective CPU cores | 9.786 | 14.392 | +47.1% |
| 32-slot CPU efficiency | 30.583% | 44.974% | +14.392 points |

The end-to-end reduction is 34.1%. It is entirely localized to atomic
publication: planning, scientific computation, and the required access-aware
storage transform remain matched. Within authority stamping, delivered read
characters fell from 45.319 GB to 3.877 GB. Descriptor stamping itself now
takes less than half a second; the largest remaining authority subphase is the
legitimate body-frame publication at 103.234 seconds.

The new output is complete, `bound_canonical_v2`, selector-ineligible, and
registry-inactive. It retains the exact source-binding digest, row-identity
digest, placement-array digest, storage profile, bundle identity, and numeric
projection digest of the control. Run-local temporal and derivation record
digests differ as designed because their sealed record references contain the
new output run ID.

## Validation completed

- `62 passed` in the complete affected atomic-publisher and subject-shape
  suites in 425.92 seconds.
- The real-Zarr
  `test_recording_bundle_publishes_coordinate_bound_members_and_subject_shape_v5`
  boundary passed in 208.19 seconds.
- Production-scale Cam2010094 read-only plans completed successfully in 18.73
  seconds at the initial checkpoint and 20.646 seconds at `e35714b0`, without
  creating scratch state or mutating the archive.
- The strengthened receipt-v2 regression passed in 189.94 seconds and checks
  every declared final array digest against the live final array.
- `py_compile` and `git diff --check` passed for the modified boundary.
- The real writer-to-publisher boundary passed in 125.76 seconds and asserts
  exactly two bundle-source reloads per proof scope.
- The 21 focused coordinate-publication and access-aware candidate tests passed
  in 356.80 seconds.
- `py_compile`, Ruff, and `git diff --check` passed for the authority-proof
  change.
- All 23 required CI checks passed for runtime commit `6d3d3c0d`.

## Guards and deferred work

- Metadata-only admission must fail if manifest, authority, lifecycle,
  selector, or consolidated metadata evidence is invalid.
- A regression test makes geometry loading a hard failure in the admission
  path.
- Scientific keypoint consumers still require the full coordinate-surface
  loader.
- Rebinding publication still performs exhaustive row/value equivalence.
- Deep subject-mask validators remain available and payload-reading.
- The commit-`bf058521` baseline completed successfully in 18,390.019 seconds
  (5h06m30s). Its plan took 1,403.552 seconds and its post-rename binding took
  9,748.514 seconds, providing a completed control for the optimized run.
- The replacement receipt profile is implemented at `e35714b0`. Its
  selector-ineligible optimized canary started only after the baseline ended
  and completed in 2,786.824 seconds, so the two runs did not contend for
  storage or CPU resources.
- Operation-scoped authority proof reuse reduced `authority_stamping` to
  117.417 seconds. The remaining 103.234-second body-frame authority step is
  now the next measured publication target; it must not be optimized by
  dropping its scientific or coordinate checks.
- The canary's degenerate-mask ellipse nondeterminism must be hardened and
  covered by repeated-run identity tests before scientific activation.
- Required CI for any new documentation or scientific-hardening head remains a
  merge and activation gate.

## Receipt-composed immutable loading follow-through (2026-09-03)

Status: **implemented locally at `47cd101e` on
`agent/palette/subject-shape-receipt-load-20260903`; required CI, a
commit-pinned cluster canary, and production cohort publication remain
pending**.

The receipt-v2 writer removed the dominant publication rescans, but normal
loading still repeated several facts already established by the completed
publication epoch:

- bundle admission reopened and canonicalized the raw and refined worker
  evidence sidecars;
- subject-shape identity and acquisition-frame arrays were decoded again by
  row-identity and temporal-authority binders;
- body-frame source, support, axis, and validity arrays were decoded repeatedly
  by nested coordinate binders; and
- the closing operation proof decoded `instance_key` once more after the
  receipt scope had ended.

The follow-through keeps one consumer interface and changes the proof source,
not the scientific contract:

1. Normal bundle admission now consumes the recording-assembly and assignment
   evidence already sealed by the core and bundle manifests. The existing
   explicit candidate/deep validator still reopens the sidecars and proves the
   producer join before a new bundle can become complete.
2. Shared row-identity, temporal-authority, coordinate-descriptor, estimator,
   and body-frame binders accept array digests only inside an explicit
   validated-scientific-receipt scope. The bound identity and temporal objects
   retain that scoped evidence for their operation-closing proof recheck.
3. Writer authority stamping does **not** enter that scientific-receipt scope.
   It still decodes identity, frame ranges, body estimator inputs, body axes,
   tail sample coordinates, and heading-formula outputs when first creating
   the authority. Digest reuse during stamping remains only a hashing
   optimization.
4. Normal loading validates exact array inventory, dtype, shape, canonical
   digest, immutable lifecycle, manifest binding, coordinate records, and
   source authority without decoding the subject-shape payload. Explicit deep
   audit still rehashes the physical payload and revalidates tail-axis,
   heading-formula, and duplicated row-identity values.

### Local production-scale evidence

A read-only trace used the selected 2,745,488-row Cam2010094 publication
`subject_shape_sleepyfish_2026_08_06_component_area_support_20260831_v002_sleepyfish_cam2010094`.
The trace replaced Zarr array access with a hard failure for every array below
that subject-shape run and then invoked the ordinary unpatched canonical
loader.

| Measurement | Result |
|---|---:|
| subject-shape array decodes during ordinary load | 0 |
| elapsed wall time | 19.64 s |
| user time | 16.48 s |
| system time | 4.36 s |
| process CPU | 106% |
| peak RSS | 1,415,028 KiB |
| exit status | 0 |

This is a workstation read-only measurement against the mounted publication,
not a cluster performance canary. It did not create scratch state, mutate a
Zarr, change a selector, or write the registry. The still-high peak RSS and
roughly 20-second metadata/source-authority traversal remain a bounded future
optimization target; they no longer include any subject-shape scientific-array
decode.

### Local validation

- 95 shared coordinate-identity and coordinate-frame tests passed.
- 20 subject-shape coordinate-publication tests passed.
- 93 eye-angle and tail-kinematics materializer tests passed.
- The real subject-mask writer -> bundle publisher -> subject-shape publisher
  -> unpatched selected reader regression passed and asserts zero array reads
  below the selected subject-shape run.
- A receipt-bound identity mutation test now states the lifecycle boundary
  explicitly: ordinary immutable loading accepts the unchanged sealed receipt,
  while `deep_audit_subject_shape_payload_receipt` detects the out-of-band
  payload mutation.
- `py_compile`, Ruff, `git diff --check`, and the Palette Git preflight passed.

Required CI remains a hard merge, deployment, activation, and production
publication gate for this follow-through commit.
