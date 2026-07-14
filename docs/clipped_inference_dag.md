# Registry-pinned clipped inference DAG

`fisheye.cluster.clipped_inference` composes one immutable LSF workflow for a
clipped recording cohort. It covers detection, detection refinement, finalized
collection creation, flat ROI cache construction, proxy crop binding,
keypoints, keypoint refinement, subject-mask inference, refined subject masks,
content validation, registry reconciliation, and optional NRS cleanup.

The target manifest schema is `palette.clipped_inference_targets.v1`. Every
target must pin its registry `recording_id`, recording directory, and canonical
analysis Zarr. The command also requires exact registry set and run identifiers
for detection, pose, and subject-mask models. Planning verifies the registered
paths and model SHA-256 values and refuses output collisions.

If a campaign stops after its detection artifacts were atomically imported,
`--resume-existing-detections` permits a new immutable run root to reuse those
groups without rerunning YOLO. The run label must remain unchanged so the
planned detection identities remain unchanged. Planning validates every import
receipt and run-group tree (excluding the ephemeral source tarball), requires a
complete run, and compares the embedded model, workflow, recording, clip,
camera, video, target, and run-name provenance to the new plan. Every later
output must still be absent. The DAG then submits short CPU revalidation jobs
in place of GPU inference jobs and continues through detection refinement.

## Execution contract

All production work is represented as structured `LsfJob` objects and wrapped
by the durable LSF runtime-status envelope. `--dry-run` writes `plan.json`,
`lsf_plan.json`, and per-target detection plans without calling `bsub`.
`--apply` performs the same planning on the workstation and sends only the
individual `bsub` commands through `login1-citrus-poller` (or
`--submit-host`). No inference, finalization, validation, or Zarr mutation runs
on the workstation or login poller.

For each recording, the dependency structure is:

```text
22 detect -> 22 detect refine -> finalized detection collection
                                      |
                               6 cache bundles
                                      |
                              proxy crop binding
                                /             \
                       22 keypoints       22 subject masks
                              |                  |
                    keypoint finalize           |
                              |                  |
                     keypoint refine            |
                               \                /
                           22 mask packages
                                  |
                         refined-mask import
                                  |
                       exact content validation
```

The cache is the physical crop materialization. The proxy crop runs bind its
rows to the canonical Zarr; there is no redundant standalone crop-image write.
Keypoints and masks safely read the same immutable cache concurrently. Each
mask package waits for both its clip-local probability shard and the exact
recording-level refined-keypoint run used for left/right eye assignment.

Detection and keypoint model outputs use the repository's indexed-sharding
defaults. Raw subject-mask probabilities are `uint8` encoded probabilities
with 32-row inner chunks and 2,048-row storage shards. Refined subject masks
are imported as authoritative dense binary `uint8 masks_roi`, with sampled
contours and eye geometry enabled and full ragged contours disabled.

The imported-run validator reports top-level success as `status=ok`; nested
checks report `status=pass`. Detection work-unit orchestration accepts both
success vocabularies so it does not reject an otherwise valid atomic import.

Validation requires all planned collection/cache/run identities, modern unique
`instance_key` arrays, equal row counts across refined detections, caches,
keypoints, and refined masks, non-binary/nonempty encoded raw probabilities,
binary/nonempty dense refined masks, sampled contours, and exact refined
keypoint assignment lineage. A single registry writer reconciles all targets
only after every validation succeeds. NRS cache and package directories are
removed only after that registry report passes its integrity check.

## Sleepyfish dry run, 2026-07-14

The reviewed three-recording plan covers cams 2010093, 2010094, and 2010096,
22 clips each. It pins:

- Detection set `detect_all_available_detect_training_v003`, run
  `detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1`.
  This keeps the new cohort consistent with completed cam2010095 and avoids
  camera-dependent automatic ranking.
- Pose set `pose_all_registry_reviewed_v2_keypoints_20260520_v001`, run
  `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`.
- Subject-mask set
  `subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001`,
  run `subject_masks_union_all_components_v001`.

The dry run rendered 368 LSF jobs: 122 jobs per target, one all-target registry
job, and one NRS cleanup job. The audit confirmed 66 clips, exact two-way mask
package joins, shared proxy gates for the parallel keypoint/mask branches,
direct cache execution with no nested `bsub`, runtime wrapping on every job,
and no submission evidence. The reviewed snapshot is under
`/tmp/sleepyfish_clipped_full_20260714_v001_dryrun`; it is diagnostic evidence,
not a durable campaign directory.
