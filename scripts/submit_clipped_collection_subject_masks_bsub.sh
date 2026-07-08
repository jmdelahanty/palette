#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_clipped_collection_subject_masks_bsub.sh --zarr PATH --collection-id ID --cache-dir-root PATH (--all-clips | --clip-id ID...) (--dry-run | --apply) [options]

Plan the clipped-collection subject-mask DAG:
  cache/proxy readiness -> per-clip subject-mask shards -> merged proxy crop run
  -> finalized refined subject masks.

Dry-run mode resolves existing clipped flat ROI cache manifests, deterministic
proxy crop run names, deterministic subject-mask shard run names, the finalizer
run name, and prints the LSF command templates. Apply mode creates proxy crop
runs on the submit host, submits subject-mask shard jobs, parses LSF job IDs,
then submits the finalizer job with a real done(<jobid>) dependency.

Common options:
  --zarr PATH                       Analysis Zarr archive
  --collection-id ID                Finalized clipped refined-detect collection id
  --cache-dir-root PATH             Root containing *.flat_roi_cache.json manifests
  --all-clips                       Plan every clip in the finalized collection
  --clip-id ID                      Plan one clip; repeatable
  --run-id ID                       Stable run id; default UTC timestamp
  --run-label LABEL                 Stable label used in run/job names
  --components NAME...              Finalized components (default: subject_body eyes_union swim_bladder)
  --assignment-keypoints-run RUN    Required when components include eyes_union
  --batch-size-sm N                 Subject-mask inference batch size (default: 128)
  --queue NAME                      Subject-mask shard LSF queue (default: gpu_l4)
  --ncores N                        Subject-mask shard CPU slots (default: 8)
  --mem-gb N                        Subject-mask shard memory GB (default: 32)
  --gpus N                          Subject-mask shard GPU count (default: 1)
  --finalizer-queue NAME            Finalizer queue (default: short)
  --finalizer-ncores N              Finalizer CPU slots (default: 8)
  --finalizer-mem-gb N              Finalizer memory GB (default: 32)
  --mask-storage MODE               Refined storage (default: dense_and_bitpacked)
  --log-dir PATH                    LSF/progress log directory
  --plan-json PATH                  Also write the plan as JSON
  --json                            Print JSON instead of text
  --dry-run                         Plan only; no jobs are submitted
  --apply                           Create proxies and submit LSF jobs
  -h, --help                        Show this message

All additional options are forwarded to
fisheye.utils.plan_clipped_collection_subject_masks_bsub.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

scripts/py -m fisheye.utils.plan_clipped_collection_subject_masks_bsub "$@"
