#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: submit_whole_recording_keypoints_bsub.sh --manifest PATH --run-label LABEL --run-root PATH --model-set-id ID --model-run-id ID (--dry-run | --apply) [options]

Plan one exact keypoint prediction job and one dependent refinement job per
manifest target, followed by one serial registry finalizer. The target manifest
must use schema palette.whole_recording_keypoint_targets.v1 and bind each
recording to one analysis Zarr and one complete flat ROI-cache manifest.

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

Defaults match the current whole-recording keypoint production profile:
traditional_v2, batch size 256, tensor input, gpu_l4 prediction with one GPU,
flat-cache staging to job-local scratch, a minimum zebrafish ROI size of
348x348, and CPU refinement on short.

Keypoint storage:
  --keypoint-roi-shard-rows N       ROI outer shard rows (default: 131072)
  --keypoint-frame-shard-rows N     Frame outer shard rows (default: 131072)
  --no-keypoint-sharding            Use ordinary chunks for keypoint outputs

All additional options are forwarded to
fisheye.cluster.keypoints.whole_recording.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

scripts/py -m fisheye.cluster.keypoints.whole_recording "$@"
