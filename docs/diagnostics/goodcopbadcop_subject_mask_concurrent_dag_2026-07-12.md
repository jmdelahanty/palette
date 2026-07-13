# GoodCopBadCop Subject-Mask Canary And Concurrent DAG

**Date:** 2026-07-12
**Status:** recording-level segmentation canary and first concurrent keypoint /
subject-mask fork-join canary complete

## Completed recording canary

Recording:

```text
2026-06-21T22-03-24Z_arena_1_GoodCopBadCop
```

Analysis archive:

```text
/groups/johnson/johnsonlab/jeremy/recordings/
  2026-06-21T22-03-24Z_arena_1_GoodCopBadCop/zarr/
  2026-06-21T22-03-24Z_arena_1_GoodCopBadCop_analysis.zarr
```

Exact runs:

```text
refined_keypoints_runs/
  refined_keypoints_goodcopbadcop_kpt5_traditional_v2_20260712_canary_v002

subject_mask_runs/
  subject_masks_unet_registry_goodcopbadcop_sm_20260712_canary_v001

refined_subject_masks_runs/
  refined_subject_masks_smart_finalizer_goodcopbadcop_sm_20260712_canary_v001
```

LSF jobs:

| Stage | Job | Result |
| --- | ---: | --- |
| U-Net inference and raw publication | `153068826[1]` | `DONE` |
| refined-mask finalization and publication | `153068827[1]` | `DONE` |

No workload ran on the login node. The jobs were submitted and monitored
through the Citrus login poller; GPU and CPU work ran on compute nodes.

## Performance

The U-Net subprocess processed `92,557` ROI rows in `1,094.41 s`, or
`84.57 rows/s`. The recording-level workflow took `1,145.00 s`; the raw publish
phase took `48.15 s`, including `28.84 s` to copy the completed run to PRFS.

This is not a material throughput regression relative to the earlier
double-buffered sharded full-clip canary:

| Run | Rows | Raw-stage seconds | Rows/s |
| --- | ---: | ---: | ---: |
| Sleepyfish double-buffered shard canary | `54,000` | `628.2` | `85.96` |
| GoodCopBadCop recording canary | `92,557` | `1,094.41` | `84.57` |

At the earlier normalized rate, the current run would take `1,076.8 s`; the
observed difference is about `17.6 s` or `1.6%`. The apparently long total time
comes primarily from processing `1.71x` as many rows.

The older regular-chunk canary reached about `106 rows/s`. The selected sharded
writer deliberately performs a fail-closed decoded destination reread/digest.
In the earlier `54,000`-row canary that terminal validation cost `150.5 s`;
direct two-buffer shard construction itself was not slower. Replacing the full
decoded reread with an equally fail-closed shard-level integrity mechanism is
the next raw-inference performance opportunity.

Refined-mask finalization took `762.33 s` inside the subprocess and
`829.11 s` for the complete staged/published workflow. Core finalization
reported `122.49 rows/s`. Sampled eye contours took `16.45 s`; sampled
body/swim postcompute took `8.77 s`. Refined publication copied `8,659` files
and `194,177,238` apparent bytes in `57.60 s`, with a `63.08 s` publish phase.

## Storage and content validation

Raw probabilities are:

```text
shape:        [92557, 3, 512, 512]
dtype:        uint8
inner chunk:  [32, 1, 512, 512]
outer shard:  [2048, 1, 512, 512]
write mode:   double_buffered_direct
buffers:      2
```

The complete raw run occupies about `359 MB` apparent storage and has `138`
probability payload objects. Thirty-two rows sampled uniformly from the first
through last row contained nonzero probabilities in every component and every
sampled row; each channel spanned the encoded range `0..255`.

Whole-run raw `mask_present` counts:

| Component | Present rows |
| --- | ---: |
| `subject_body` | `92,557` |
| `eyes_union` | `92,557` |
| `swim_bladder` | `92,288` |

The authoritative refined surface is physically present:

```text
masks_roi shape: [92557, 4, 512, 512]
dtype:           uint8
chunk:           [256, 1, 512, 512]
authority:       dense masks_roi
```

All four refined components contained binary nonzero pixels in all `32`
uniformly sampled rows. Whole-run refined `mask_present` counts:

| Component | Present rows |
| --- | ---: |
| `subject_body` | `92,557` |
| `eye_left` | `91,606` |
| `eye_right` | `91,606` |
| `swim_bladder` | `92,029` |

Fixed-K sampled contour groups are present for body, left eye, right eye, and
swim bladder. Full ragged `contours` groups are absent. Sample counts are body
`128`, each eye `64`, and swim bladder `32`. The run records
`component_contours_requested=false` and
`sampled_component_contours_requested=true`.

Registry `recording_step_status_latest` reports both `subject_masks` and
`refined_subject_masks` as `ok` for the exact runs above.

## First concurrent DAG canary

The first complete fork-join run used the simultaneous neighboring recording:

```text
2026-06-21T22-03-24Z_arena_2_GoodCopBadCop
```

The immutable flat ROI cache contained `136,200` rows and
`35,704,012,800` bytes. Job `153068898` built it in `725.92 s`
(`187.62 rows/s`) and published the payload to NRS in `38.50 s` at
`884.35 MiB/s`. This was faster per retained ROI than the arena-1 cache
(`128.30 rows/s`) because both recordings decoded about `139,596` frames while
arena 2 retained more ROIs.

The concurrent workflow jobs were:

| Stage | Job | Result | Runtime |
| --- | ---: | --- | ---: |
| keypoint inference and publication | `153068925` | `DONE` | `502 s` |
| subject-mask inference and publication | `153068926` | `DONE` | `1,768 s` |
| keypoint refinement | `153068927` | `DONE` | `70 s` |
| refined-mask finalization and publication | `153068928` | `DONE` | `888 s` |
| serial registry reconciliation | `153068929` | `DONE` | `20 s` |

Both inference roots ran concurrently on `h08u16` and staged independent copies
of the same cache. Their cache/startup intervals were about `106-111 s`, or
roughly `325 MiB/s` per job and `650 MiB/s` aggregate. The concurrent reads
therefore cost each job about `70 s` relative to the cache publisher's single
copy, but did not serialize the roots.

Keypoint inference processed all rows in `367.91 s` (`370.19 rows/s`). It
produced `135,223` successful rows and `977` unsuccessful rows (`99.28%`
coverage). Refinement reported `135,150` clean usable rows, `73` geometry
issues, and one corrected flip.

The U-Net subprocess processed all rows in `1,570.62 s`, or `86.72 rows/s`.
That was about `2.5%` faster than arena 1's `84.57 rows/s` despite concurrent
keypoint inference. Its complete staged validation/publish workflow took
`1,645.50 s` after cache staging.

The refined-mask subprocess completed in `779.58 s`; its progress summary
reported `176.64 rows/s` including eye geometry and sampled contours. Eye
geometry took `1.92 s`, sampled eye contours `23.58 s`, and sampled body/swim
postcompute `21.42 s`. Refined publication and published-output validation took
about `96 s`. LSF reported `15,658 MB` peak memory across the 16-worker job,
well below its `32 GB` allocation and without the old per-worker identity-map
amplification.

Independent validation job `153068980` read 32 uniformly spaced raw and refined
rows on a compute node. Every sampled row contained nonzero data in every
component. Raw probabilities spanned `0..255`; refined dense masks contained
only `0` and `1`.

Whole-run `mask_present` counts were:

| Surface | Component | Present rows |
| --- | --- | ---: |
| raw | `subject_body` | `136,200` |
| raw | `eyes_union` | `135,790` |
| raw | `swim_bladder` | `136,173` |
| refined | `subject_body` | `136,200` |
| refined | `eye_left` | `135,126` |
| refined | `eye_right` | `135,126` |
| refined | `swim_bladder` | `136,125` |

The published raw probability metadata is indexed sharding with outer shape
`[2048,1,512,512]`, inner shape `[32,1,512,512]`, and
`write_mode=double_buffered_direct` with two buffers. Keypoint ROI coordinates
are indexed shards with outer row size `65,536` and inner row size `1,024`.
Refined dense masks are regular `[256,1,512,512]` chunks.

All four sampled-contour groups are present with `128` body points, `64` points
per eye, and `32` swim-bladder points. Full ragged contour groups are absent.
The refined contract validator returned no errors or warnings and required no
backfill. The serial registry finalizer recorded the exact refined-keypoint
assignment lineage, reconciled all four stages as `ok`, and finished with
`PRAGMA integrity_check = ok`.

## Pre-July Wave 1 NRS quota incident

The first eight-recording pre-July wave built eight NRS flat ROI caches totaling
about `260 GiB`. All keypoint inference and refinement jobs completed. All eight
subject-mask inference workers also completed U-Net inference, exact sharded
destination digest validation, atomic PRFS publication, and Zarr completion
marking, but their wrappers then exited while creating optional NRS handoff
tars with `[Errno 122] Disk quota exceeded`. The dependent mask finalizers and
serial registry finalizer therefore never started.

The first recording's published raw run contains `143,305` rows and is formally
marked complete. Independent compute-node validation job `153069845` found
nonzero `0..255` probabilities in all three channels across 32 uniformly spaced
rows. Whole-run raw `mask_present` counts were `143,305` body, `143,305`
eyes-union, and `143,295` swim bladder. Wave 1 can therefore be recovered by
running only mask finalization and registry reconciliation; GPU inference does
not need to be repeated.

After confirming no unfinished Wave 1 jobs, the exact eight-cache directory
`goodcopbadcop_prejuly_wave01_20260712_01` was removed from NRS, freeing about
`260 GiB`. Durable plans, logs, and published Zarr runs were retained.

Wave 1 recovery reused the published raw masks and refined keypoints. Eight
CPU-only mask finalizers (`153069866` through `153069873`) completed in
`537-900 s`; no inference was repeated. Dependent registry reconciler
`153069874` completed in `89 s`, reconciled all eight exact run families, and
reported `PRAGMA integrity_check = ok`.

Read-only compute-node validation job `153069884` then checked all eight dense
refined outputs. Every run was complete with exact refined-keypoint assignment
lineage, no contract errors or warnings, binary nonzero samples in every
component across 32 uniformly spaced rows, fixed-K sampled contours, and no
full ragged contours. Across `1,064,268` refined rows, whole-run presence counts
were `1,064,268` body, `1,048,673` for each eye, and `1,020,747` swim bladder.
The validation report had `error_count=0`.

## Pre-July Wave 2 completion

Wave 2 reduced recording concurrency to four to keep the immutable flat ROI
caches within NRS quota. Cache jobs `153069897-153069900` built `134 GiB`
across four recordings. Keypoint inference `153070651-153070654`, subject-mask
inference `153070655-153070658`, keypoint refinement
`153070659-153070662`, and mask finalization `153070663-153070666` all
completed. Subject-mask inference took `1,722-1,814 s`; finalization took
`447-632 s` and peaked at `13.7-14.5 GiB` per recording. The finalizer stderr
contained only expected Zarr v3 consolidated-metadata and null-terminated-byte
portability warnings.

Registry reconciler `153070667` reported `status=ok`, reconciled all four exact
run families, and finished with `PRAGMA integrity_check = ok`. Separate cleanup
job `153070668` then deleted `143,771,828,224` cache bytes and removed the
Wave 2 NRS cache root.

Read-only compute-node validation job `153070737` checked the four dense refined
outputs and completed with `error_count=0`. Every run was complete, its formal
contract had no errors or warnings, and its exact refined-keypoint assignment
lineage matched the Wave 2 plan. Across `548,446` refined rows, whole-run
presence counts were `548,446` body, `536,844` for each eye, and `548,196` swim
bladder. Thirty-two uniformly spaced rows per recording contained binary
`uint8` values `{0,1}` and nonzero pixels in every component. Fixed-K sampled
contours were present at `128` body, `64` per eye, and `32` swim-bladder points;
eye ellipse geometry was populated, and full ragged contour groups were absent.

The first validation submission, job `153070731`, read the canonical
`assignment_keypoints_run` provenance attribute using an incorrect singular
spelling in the one-off checker. It therefore exited nonzero even though every
mask and contract check passed. Job `153070737` corrected only that checker key
and is the authoritative Wave 2 validation result.

## Pre-July Wave 3 cleanup repair

Wave 3 processed four recordings and its combined registry reconciler
`153071579` completed successfully. The original terminal cleanup job
`153071984` failed closed before deleting anything because the early combined
analysis plan omitted `roi_cache_payload`, although its immutable nested
keypoint plan retained each exact cache payload path, manifest SHA-256, crop
identity, shape, and byte count. The cache root remained intact at `136 GiB`
with four manifests and four payloads.

The cleanup repair wrote a separate plan from those immutable keypoint-plan
bindings and did not mutate the original analysis plan. Cleanup-only job
`153074138` then passed preflight, deleted all eight artifacts payloads-first,
and completed `DONE`. Its report records four caches and
`145,037,745,235` total bytes deleted. The Wave 3 cache root
`goodcopbadcop_prejuly_wave03_20260713_01` is absent. The durable repaired plan
and cleanup report are under the Wave 3 run's `cleanup/` directory.

## Concurrent fork-join contract

Raw U-Net subject-mask inference does not consume keypoints. Only refined-mask
finalization needs them, to split `eyes_union` into anatomically assigned left
and right eyes. The safe recording-level DAG is therefore:

```text
optional cache build/publish
                |
flat ROI cache ─┬─> keypoint inference -> keypoint refinement ─┐
                └─> subject-mask inference --------------------┤
                                                              v
                                                  mask finalization
                                                              |
                                                output validation
                                                              |
                                                   registry reconciliation
                                                              |
                                             optional NRS cache cleanup
```

Required invariants:

1. The ROI cache is immutable and complete before either inference job starts.
   It may be validated at planning time or produced by an exact upstream cache
   job in the same DAG.
2. Each inference job stages its own cache copy to node-local scratch. This is
   safe even when both read the same durable cache concurrently; PRFS bandwidth
   contention remains a performance variable for the first combined canary.
3. Keypoint and mask inference write distinct run groups through private staged
   outputs and atomic publication.
4. Raw mask inference carries no assignment-keypoint binding.
5. Mask finalization has two `done(...)` dependencies: the exact mask-inference
   job and the exact keypoint-refinement job for the same target.
6. The finalizer receives the deterministic refined-keypoint run name explicitly
   and never resolves `latest`.
7. Full ragged contours remain opt-in. Eye geometry and sampled contours are
   production defaults.
8. Cache cleanup is a separate `done(registry_finalize)` job. Existing caches
   retain their immutable plan-time digest. Caches built inside the DAG are
   late-bound and are instead validated against their planned manifest path,
   payload path, source analysis Zarr, crop run, format, completeness flag, and
   allowed NRS root before deleting any payload.
9. Independent validation depends on every exact mask finalizer and runs before
   registry reconciliation. It checks the exact keypoint and mask runs,
   assignment lineage, raw encoded probabilities, dense refined masks,
   `mask_present`, sampled contours, full-contour absence, and the refined mask
   contract.

## Implementation

The first implementation is:

- `fisheye.cluster.whole_recording_analysis`: composes reusable `roi_cache`,
  `keypoints`, `subject_masks`, `analysis_validation`, `registry`, and optional
  `cache_cleanup` fragments into one validated LSF workflow;
- `fisheye.cluster.whole_recording_analysis_validate`: independently samples
  exact raw and refined mask outputs and fails closed before registry promotion;
- `fisheye.cluster.flat_roi_cache`: builds on node-local scratch and atomically
  publishes the payload before the manifest. With `--roi-cache-policy build`,
  both inference branches depend on its per-target producer job;
- `fisheye.cluster.subject_masks.recording`: stages the immutable flat cache on
  the inference node and invokes one split subject-mask workflow stage;
- all keypoint and mask workers defer registry mutation; one combined fan-in
  finalizer validates exact run completion and assignment lineage, reconciles
  every stage serially, refreshes subject-mask registry views, and finishes
  with `PRAGMA integrity_check`;
- `fisheye.cluster.whole_recording_analysis_cache_cleanup`: optional terminal
  LSF job enabled with `--cleanup-roi-caches-after-success`. It depends on
  `done(registry_finalize)`, verifies either the immutable plan-time digest or
  the late-bound source/crop identity plus the exact payload binding under the
  allowed NRS cache root before deleting anything, deletes payloads before
  manifests, and writes an independent cleanup report;
- `run_subject_mask_batch_pipeline`: accepts an exact
  `--assignment-keypoint-group` plus `--assignment-keypoints-run` override and
  fails closed if that run is absent;
- `scripts/submit_whole_recording_analysis_bsub.sh`: thin entry point for a
  durable dry-run/apply plan bundle.

The composed planner writes the exact DAG and target bindings before submission.
Do not reuse completed run names from this note because output collisions fail
closed.

A planning-only smoke using the reviewed manifest
`goodcopbadcop_keypoints_20260712_canary.json` passed against the real registry,
cache manifest, crop run, model artifact, and analysis Zarr. It wrote a temporary
five-job plan and submitted no work. Its dependency order was:

```text
predict ─────> refine ───────┐
mask_infer ──────────────────┴─> mask_finalize -> validate -> registry_finalize -> cache_cleanup
```

`cache_cleanup` is deliberately a separate job rather than part of registry
finalization. An upstream failure prevents deletion; a cleanup failure does not
invalidate already reconciled analysis outputs. The remaining pre-July
GoodCopBadCop waves should enable this job and should omit optional NRS handoff
tar creation, because the published PRFS run groups are the durable outputs.

The generated mask-finalizer dependency was exactly
`done(mask_infer) && done(refine)`, and its worker command named the planned
`refined_keypoints_<run-label>` explicitly.

The first combined canary compared:

- end-to-end wall time against serial keypoints then segmentation;
- per-job ROI-cache staging time and whether the two concurrent PRFS reads
  reduce either inference rate;
- exact refined-keypoint lineage in the refined-mask attrs;
- nonzero raw/refined mask samples and whole-run `mask_present` counts;
- sampled-contour presence and full-contour absence;
- registry integrity and exact final step rows.

## Remaining pre-July campaign plan

Registry reconciliation job `153075623` repaired the production registration
for `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop` on compute host `h07u19`.
The production archive is now the canonical source-recording dataset
`2026-06-14T21-12-08Z_arena_4:z13152d64687b`; the JLCRSI example copy remains
a distinct derived-analysis dataset.

The remaining campaign contains 18 production recordings. A real metadata-only
dry-run planned 93 jobs: 18 cache builds, 18 keypoint inference jobs, 18
keypoint refinements, 18 subject-mask inference jobs, 18 mask finalizers, one
independent validator, one serial registry reconciler, and one terminal cache
cleanup. The planned flat ROI payload total is `626,653,396,992` bytes, well
within the 5 TB `/nrs/johnson/` share.

The campaign uses `--max-active-targets 8`. The first eight cache jobs are DAG
roots; each later cache build has a rolling dependency on the mask finalizer
eight targets earlier. Validation depends on all 18 exact mask finalizers,
registry reconciliation depends on validation, and cache cleanup depends on
registry success. Cache paths are confined to:

```text
/nrs/johnson/palette_staging/flat_roi_cache/
  goodcopbadcop_prejuly_remaining_20260713_01/
```
