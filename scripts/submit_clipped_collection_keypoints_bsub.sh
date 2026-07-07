#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_clipped_collection_keypoints_bsub.sh --zarr PATH --collection-id ID --cache-dir-root PATH (--all-clips | --clip-id ID...) (--dry-run | --apply) [options]

Plan the clipped-collection keypoint DAG:
  cache/proxy readiness -> per-clip keypoint shards -> merged proxy crop run
  -> finalized keypoints -> refined keypoints.

Dry-run mode resolves existing clipped flat ROI cache manifests, deterministic
proxy crop run names, deterministic keypoint shard run names,
finalizer/refinement run names, and prints the LSF command templates. Apply
mode creates proxy crop runs on the submit host, submits keypoint shard jobs,
parses LSF job IDs, then submits finalizer/refinement jobs with real
done(<jobid>) dependencies.

Common options:
  --zarr PATH                       Analysis Zarr archive
  --collection-id ID                Finalized clipped refined-detect collection id
  --cache-dir-root PATH             Root containing *.flat_roi_cache.json manifests
  --all-clips                       Plan every clip in the finalized collection
  --clip-id ID                      Plan one clip; repeatable
  --run-id ID                       Stable run id; default UTC timestamp
  --run-label LABEL                 Stable label used in run/job names
  --pose-schema NAME                Pose schema (default: traditional_v2)
  --batch-size-kp N                 Keypoint inference batch size (default: 256)
  --queue NAME                      Keypoint shard LSF queue (default: gpu_l4)
  --ncores N                        Keypoint shard CPU slots (default: 4)
  --mem-gb N                        Keypoint shard memory GB (default: 32)
  --gpus N                          Keypoint shard GPU count (default: 1)
  --finalizer-queue NAME            Finalizer queue (default: normal)
  --refine-queue NAME               Refined-keypoint queue (default: normal)
  --log-dir PATH                    LSF/progress log directory
  --plan-json PATH                  Also write the plan as JSON
  --json                            Print JSON instead of text
  --dry-run                         Plan only; no jobs are submitted
  --apply                           Create proxies and submit LSF jobs
  -h, --help                        Show this message

All additional options are forwarded to
fisheye.utils.plan_clipped_collection_keypoints_bsub.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

scripts/py -m fisheye.utils.plan_clipped_collection_keypoints_bsub "$@"
