# Subject-shape planning admission optimization — 2026-08-29

Status: **implemented, locally validated, and committed as `f8f8abfa` in the
isolated optimization worktree; not merged or production-active**.

Worktree branch: `agent/palette/subject-shape-plan-receipts-20260829`

Base/canary commit: `bf058521`

This optimization follows the shared receipt lifecycle decision in
[`publication_receipt_hashing_lifecycle_2026-08-29.md`](publication_receipt_hashing_lifecycle_2026-08-29.md).
It improves input admission and planning. It does not yet replace the
subject-shape writer's post-compute payload scan with write-time receipts.

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
| command wall time | 677.871 s | 18.73 s |
| plan telemetry wall time | not separately retained | 17.494 s |
| delivered read characters | more than 26 GB observed from `/proc` | 1,184,091,249 bytes |
| peak RSS | not retained | 917,776 KiB |
| result | valid plan | valid plan with the same source binding |

The current worktree is approximately **36.2 times faster**, a **97.2% wall-time
reduction** from the reproduced partial-optimization baseline. The plan retained
the same bundle, rebinding, row count, source-binding digest, coordinate
authority digest, and member-manifest digests.

The remaining approximately 1.18 GB of delivered metadata reads are visible and
should be investigated separately. They are no longer a reason to block the
four-camera analytics delivery, but root-metadata parsing and declaration-map
construction remain candidates for a later bounded optimization.

## Validation completed

- `61 passed` in the complete affected subject-mask core, cache, quality,
  subject-position keypoint source, and assignment-rebinding unit suites.
- The real-Zarr
  `test_recording_bundle_publishes_coordinate_bound_members_and_subject_shape_v5`
  boundary passed in 208.19 seconds.
- A production-scale Cam2010094 read-only plan completed successfully in 18.73
  seconds without creating scratch state or mutating the archive.
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
- The running commit-`bf058521` canary still performs its post-compute decoded
  payload scan; this worktree cannot alter an already-submitted immutable job.
- Replacing that scan requires the separately documented write-time,
  chunk-aligned receipt profile and compatibility migration.
