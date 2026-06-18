# Subject-Mask Cluster Submission Audit - 2026-06-18

## Summary

Update after implementation: `scripts/submit_subject_mask_batches_bsub.sh` now
provides the modern subject-mask LSF submission surface. The original audit
below is retained to document why this wrapper exists and what it replaces.

The modern subject-mask runtime path exists, and the cluster submission surface
now targets it directly instead of routing through the deprecated eye-mask
submitter.

Current modern path:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>
```

Serial wrapper:

```bash
scripts/run_subject_mask_batch_pipeline
```

That wrapper calls:

```bash
scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline ...
```

It remains useful as a conservative serial driver. The LSF entry point is now:

```bash
scripts/submit_subject_mask_batches_bsub.sh
```

## What Exists Today

`src/fisheye/utils/run_subject_mask_batch_pipeline.py`:

- discovers `*_analysis.zarr` targets from filesystem roots or from a previous
  JSON report
- resolves the subject-mask model from the registry
- chooses latest crop and keypoint/refined-keypoint runs by zarr metadata
- runs U-Net subject-mask inference
- runs smart refined-subject finalization
- validates that `subject_mask_runs/<run>` and
  `refined_subject_masks_runs/<run>` exist
- can write JSON and Markdown reports
- runs archives serially inside one process

`src/fisheye/segmentation/infer_unet_subject_masks.py`:

- supports `--resolve-model-from-registry`
- records model-resolution provenance on the run
- records ROI cache/source details
- records platform provenance, including hostname through `get_environment_info`
- emits `recording_step_status` for `subject_masks`

`src/fisheye/refinement/finalize_subject_masks.py`:

- writes canonical `refined_subject_masks_runs/<run>`
- emits `recording_step_status` for `refined_subject_masks`

`src/fisheye/utils/system.py`:

- captures LSF fields when run under LSF, including job id, job index, queue,
  requested GPU string, host list, and `CUDA_VISIBLE_DEVICES`

## Implemented LSF Wrapper

`scripts/submit_subject_mask_batches_bsub.sh` now:

- uses registry discovery by default
- submits one LSF array task per selected recording
- resolves per-recording flat ROI cache manifests from `--roi-cache-root`
- optionally accepts `--roi-cache-manifest` for a single selected recording
- stages the cache manifest and `.bin` payload to node-local scratch by default
- passes the staged manifest into subject-mask inference
- uses a shell `EXIT` trap to delete the staged local cache on success or
  failure
- writes per-recording JSON and Markdown reports under the submit run directory

The existing `scripts/submit_eye_masks_batches_bsub.sh` is for the older
eye-mask stage family. It should not be treated as the modern subject-mask
submission interface.

The modern subject-mask batch driver now supports registry-backed target
discovery and `--emit-paths`, so it can be used by the submitter in the same
style as the detect/crop/keypoint batch tools.

## Stale Documentation

`docs/cluster_batching_guide.md` was updated to describe stage 4 as subject
masks and point to `scripts/submit_subject_mask_batches_bsub.sh`.

`docs/operator_guide/pipeline_workflow.md` still shows `/nvme1/palette_registry.sqlite`
in the subject-mask example.

These are documentation drift points. New cluster work should target
`subject_mask_runs` and `refined_subject_masks_runs`, not `eye_masks_runs`, and
should use the PRFS registry:

```text
/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

## Follow-Up Implementation Slice

1. Run a dry-run on the login node against a small scoped subset and inspect
   `targets.tsv`.
2. Submit one recording with `--max-active 1` and verify:
   - staged local cache appears under `/scratch/$USER/$LSB_JOBID/...`
   - subject-mask inference consumes the staged manifest
   - refined-subject finalization completes
   - the local staged cache directory is removed after the job exits
3. If the one-recording run is clean, submit the target set with conservative
   concurrency such as `--max-active 2` or `--max-active 4`.
4. Consider extracting the flat-cache staging helper into a shared Python module
   so keypoints and subject masks use one implementation rather than parallel
   shell/Python staging snippets.

## Safety Notes

Use the PRFS registry and PRFS recording roots for cluster jobs. Do not submit
cluster jobs against `/nvme1` paths; compute nodes cannot see workstation-local
storage.

Start with low concurrency. Subject-mask inference writes large probability
arrays, and refined finalization writes canonical masks, metrics, reasons, and
component groups. A small `--max-active` is safer for PRFS until measured.

The eye-mask stage is deprecated for new work. Keep historical read support, but
do not build new cluster workflows around `eye_masks_runs`.
