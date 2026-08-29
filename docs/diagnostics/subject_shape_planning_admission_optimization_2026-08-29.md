# Subject-shape planning admission optimization — 2026-08-29

Status: **implemented through `e35714b0`, locally validated, pushed to draft
PR #73, and exercised by a completed selector-ineligible production-scale
canary. Required CI passed at the runtime commit. A separate pre-existing
ellipse-fit nondeterminism finding still requires scientific hardening; not
merged or production-active**.

Worktree branch: `agent/palette/subject-shape-plan-receipts-20260829`

Legacy baseline canary commit: `bf058521`

Optimized canary commit: `e35714b0`

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
- `authority_stamping` still takes 912.950 seconds and performs about 43.008 GB
  of delivered metadata reads. The physical payload hash takes only 2.912
  seconds. A later pass should eliminate repeated large-metadata
  read/modify/write serialization without weakening authority validation.
- The canary's degenerate-mask ellipse nondeterminism must be hardened and
  covered by repeated-run identity tests before scientific activation.
- Required CI for any new documentation or scientific-hardening head remains a
  merge and activation gate.
