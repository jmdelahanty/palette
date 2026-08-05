#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_whole_recording_keypoints_bsub.sh --manifest PATH --run-label LABEL --run-root PATH --model-set-id ID --model-run-id ID (--dry-run | --apply) [options]

Plan one terminal keypoint prediction job and one dependent strict-v2
finalization job per manifest target, followed by one candidate validator. The
target manifest must use schema palette.whole_recording_keypoint_targets.v1 and
bind each recording to one analysis Zarr and one complete flat ROI-cache
manifest.

Safety:
  --dry-run validates every target/cache/model, writes the plan bundle, and
  prints bsub templates without submitting anything.
  --apply is the only submission path and refuses existing output run groups or
  an existing lsf_submission.json.

Required:
  --manifest PATH          Explicit reviewed target manifest
  --run-label LABEL        Deterministic suffix for keypoint/refined run names
  --run-root PATH          Durable plan, status, progress, and LSF log directory
  --model-set-id ID        Exact registered pose-model set
  --model-run-id ID        Exact successful pose training run
  --dry-run | --apply      Explicit execution mode

Defaults match the current whole-recording keypoint candidate profile:
traditional_v2, batch size 256, tensor input, gpu_l4 prediction with one GPU,
flat-cache staging to job-local scratch, a minimum zebrafish ROI size of
348x348, and CPU finalization on short. No selector or registry activation is
performed.

Strict-v2 keypoint storage is derived from dtype, per-row shape, byte budgets,
and access class. Historical row-count shard options are accepted only as
no-effect compatibility inputs and are recorded as such in the v2 plan.

All additional options are forwarded to
fisheye.cluster.keypoints.whole_recording.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
"${repo_root}/scripts/py" -m fisheye.cluster.keypoints.whole_recording "$@"
