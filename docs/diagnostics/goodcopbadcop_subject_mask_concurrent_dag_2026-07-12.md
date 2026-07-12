# GoodCopBadCop Subject-Mask Canary And Concurrent DAG

**Date:** 2026-07-12
**Status:** recording-level segmentation canary complete; concurrent keypoint /
subject-mask fork-join planner implemented and awaiting its first cluster dry-run
and canary

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

## Concurrent fork-join contract

Raw U-Net subject-mask inference does not consume keypoints. Only refined-mask
finalization needs them, to split `eyes_union` into anatomically assigned left
and right eyes. The safe recording-level DAG is therefore:

```text
flat ROI cache ─┬─> keypoint inference -> keypoint refinement ─┐
                └─> subject-mask inference --------------------┤
                                                              v
                                                  mask finalization
                                                              |
                                                   registry reconciliation
```

Required invariants:

1. The ROI cache is immutable and complete before either inference job starts.
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

## Implementation

The first implementation is:

- `fisheye.cluster.whole_recording_analysis`: composes the existing reviewed
  whole-recording keypoint plan with mask inference/finalization jobs;
- `fisheye.cluster.subject_masks.recording`: stages the immutable flat cache on
  the inference node and invokes one split subject-mask workflow stage;
- all keypoint and mask workers defer registry mutation; one combined fan-in
  finalizer validates exact run completion and assignment lineage, reconciles
  every stage serially, refreshes subject-mask registry views, and finishes
  with `PRAGMA integrity_check`;
- `run_subject_mask_batch_pipeline`: accepts an exact
  `--assignment-keypoint-group` plus `--assignment-keypoints-run` override and
  fails closed if that run is absent;
- `scripts/submit_whole_recording_analysis_bsub.sh`: thin entry point for a
  durable dry-run/apply plan bundle.

The composed planner writes the exact DAG and target bindings before submission.
Its intended first use is a dry-run against one fresh GoodCopBadCop recording,
followed by a one-recording cluster canary. Do not reuse the completed run names
from this note because output collisions fail closed.

A planning-only smoke using the reviewed manifest
`goodcopbadcop_keypoints_20260712_canary.json` passed against the real registry,
cache manifest, crop run, model artifact, and analysis Zarr. It wrote a temporary
five-job plan and submitted no work. Its dependency order was:

```text
predict ─────> refine ───────┐
mask_infer ──────────────────┴─> mask_finalize -> registry_finalize
```

The generated mask-finalizer dependency was exactly
`done(mask_infer) && done(refine)`, and its worker command named the planned
`refined_keypoints_<run-label>` explicitly.

The first combined canary should compare:

- end-to-end wall time against serial keypoints then segmentation;
- per-job ROI-cache staging time and whether the two concurrent PRFS reads
  reduce either inference rate;
- exact refined-keypoint lineage in the refined-mask attrs;
- nonzero raw/refined mask samples and whole-run `mask_present` counts;
- sampled-contour presence and full-contour absence;
- registry integrity and exact final step rows.
