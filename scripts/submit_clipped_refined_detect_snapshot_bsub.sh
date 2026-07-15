#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
OUTPUT_ROOT="${PALETTE_CLIPPED_DETECT_SNAPSHOT_ROOT:-/groups/johnson/johnsonlab/jeremy/recordings/logs/clipped_refined_detect_snapshot_bsub}"
RUN_ID=""
ZARR_PATH=""
OUTPUT_RUN=""
COLLECTION_ID=""
FRAME_INDEX=""
KEYPOINT_RUN=""
MASK_RUN=""
QUEUE="short"
MEM_GB=8
WALLTIME="1:00"
SHARD_ROWS=131072
APPLY=0
PROMOTE=1
BACKFILL_MASK=1
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_clipped_refined_detect_snapshot_bsub.sh [options]

Render or submit one CPU job that materializes a finalized clipped refined-
detection collection as one recording-level indexed-sharded snapshot. By
default the same job also plans or applies exact-lineage instance_key backfill
to the selected refined subject-mask run. The login host only issues bsub; all
Zarr/Parquet reads and writes occur inside the LSF allocation.

Required:
  --run-id ID
  --zarr-path PATH
  --output-run NAME

Options:
  --collection-id ID          Default: refined_detect_runs.latest_collection
  --recording-frame-index PATH
  --keypoint-run NAME         Explicit key source for mask backfill
  --mask-run NAME             Explicit refined-mask target
  --shard-rows N              Default: 131072
  --skip-mask-backfill
  --apply                     Write and promote; default is dry-run
  --no-promote                Publish without changing refined-detect pointers
  --palette-repo PATH
  --output-root PATH
  --submit-host HOST
  --queue NAME                Default: short
  --mem-gb N                  Default: 8
  --walltime H:MM             Default: 1:00
  --submit                    Submit; otherwise render only
  -h, --help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;;
    --zarr-path) ZARR_PATH="$2"; shift 2;;
    --output-run) OUTPUT_RUN="$2"; shift 2;;
    --collection-id) COLLECTION_ID="$2"; shift 2;;
    --recording-frame-index) FRAME_INDEX="$2"; shift 2;;
    --keypoint-run) KEYPOINT_RUN="$2"; shift 2;;
    --mask-run) MASK_RUN="$2"; shift 2;;
    --shard-rows) SHARD_ROWS="$2"; shift 2;;
    --skip-mask-backfill) BACKFILL_MASK=0; shift;;
    --apply) APPLY=1; shift;;
    --no-promote) PROMOTE=0; shift;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$RUN_ID" ]] || fail "--run-id is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --run-id: $RUN_ID"
[[ -f "$ZARR_PATH/zarr.json" ]] || fail "not a Zarr v3 root: $ZARR_PATH"
[[ -n "$OUTPUT_RUN" && "$OUTPUT_RUN" != */* ]] || fail "unsafe --output-run: $OUTPUT_RUN"
if [[ -n "$COLLECTION_ID" ]]; then
  [[ "$COLLECTION_ID" != */* ]] || fail "unsafe --collection-id: $COLLECTION_ID"
fi
if [[ -n "$KEYPOINT_RUN" ]]; then
  [[ "$KEYPOINT_RUN" != */* ]] || fail "unsafe --keypoint-run: $KEYPOINT_RUN"
fi
if [[ -n "$MASK_RUN" ]]; then
  [[ "$MASK_RUN" != */* ]] || fail "unsafe --mask-run: $MASK_RUN"
fi
[[ "$SHARD_ROWS" =~ ^[1-9][0-9]*$ ]] || fail "--shard-rows must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable: $PALETTE_REPO"
[[ -f "$PALETTE_REPO/src/fisheye/utils/publish_clipped_refined_detect_snapshot.py" ]] || \
  fail "collection snapshot publisher is missing from: $PALETTE_REPO"
[[ -f "$PALETTE_REPO/src/fisheye/utils/backfill_refined_subject_mask_instance_keys.py" ]] || \
  fail "mask instance-key repair is missing from: $PALETTE_REPO"

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"
SNAPSHOT_REPORT="$RUN_DIR/snapshot_report.json"
MASK_REPORT="$RUN_DIR/mask_instance_key_report.json"
JOB_SCRIPT="$RUN_DIR/run_clipped_refined_detect_snapshot.sh"
STATUS_FILE="$RUN_DIR/status.txt"
SUBMISSION_FILE="$RUN_DIR/submission.txt"
EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_output="$(printf '%q' "$OUTPUT_RUN")"
q_collection="$(printf '%q' "$COLLECTION_ID")"
q_frame_index="$(printf '%q' "$FRAME_INDEX")"
q_keypoint="$(printf '%q' "$KEYPOINT_RUN")"
q_mask="$(printf '%q' "$MASK_RUN")"
q_shards="$(printf '%q' "$SHARD_ROWS")"
q_snapshot_report="$(printf '%q' "$SNAPSHOT_REPORT")"
q_mask_report="$(printf '%q' "$MASK_REPORT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
OUTPUT_RUN=${q_output}
COLLECTION_ID=${q_collection}
FRAME_INDEX=${q_frame_index}
KEYPOINT_RUN=${q_keypoint}
MASK_RUN=${q_mask}
SHARD_ROWS=${q_shards}
SNAPSHOT_REPORT=${q_snapshot_report}
MASK_REPORT=${q_mask_report}
STATUS_FILE=${q_status}
EXPECTED_COMMIT=${q_commit}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
snapshot_cmd=(
  scripts/py -m fisheye.utils.publish_clipped_refined_detect_snapshot
  "\${ZARR_PATH}"
  --output-run "\${OUTPUT_RUN}"
  --shard-rows "\${SHARD_ROWS}"
  --json
)
if [[ -n "\${COLLECTION_ID}" ]]; then snapshot_cmd+=(--collection-id "\${COLLECTION_ID}"); fi
if [[ -n "\${FRAME_INDEX}" ]]; then snapshot_cmd+=(--recording-frame-index "\${FRAME_INDEX}"); fi
JOBSCRIPT

if [[ "$APPLY" == "1" ]]; then
  printf 'snapshot_cmd+=(--apply)\n' >>"$JOB_SCRIPT"
fi
if [[ "$PROMOTE" == "0" ]]; then
  printf 'snapshot_cmd+=(--no-promote)\n' >>"$JOB_SCRIPT"
fi

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
printf 'snapshot_command='; printf '%q ' "${snapshot_cmd[@]}"; printf '\n'
"${snapshot_cmd[@]}" >"${SNAPSHOT_REPORT}"
JOBSCRIPT

if [[ "$BACKFILL_MASK" == "1" ]]; then
  cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
mask_cmd=(
  scripts/py -m fisheye.utils.backfill_refined_subject_mask_instance_keys
  "${ZARR_PATH}"
  --block-rows "${SHARD_ROWS}"
  --json
)
if [[ -n "${KEYPOINT_RUN}" ]]; then mask_cmd+=(--keypoint-run "${KEYPOINT_RUN}"); fi
if [[ -n "${MASK_RUN}" ]]; then mask_cmd+=(--mask-run "${MASK_RUN}"); fi
JOBSCRIPT
  if [[ "$APPLY" == "1" ]]; then
    printf 'mask_cmd+=(--apply)\n' >>"$JOB_SCRIPT"
  fi
  cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
printf 'mask_command='; printf '%q ' "${mask_cmd[@]}"; printf '\n'
"${mask_cmd[@]}" >"${MASK_REPORT}"
JOBSCRIPT
else
  printf ': >"${MASK_REPORT}"\n' >>"$JOB_SCRIPT"
fi

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'snapshot_report=%s\n' "${SNAPSHOT_REPORT}"
  printf 'mask_report=%s\n' "${MASK_REPORT}"
} >"${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "clipped_detect_snapshot_${RUN_ID}"
  -n 1
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G] span[hosts=1]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'operation=%s\n' "$([[ "$APPLY" == "1" ]] && printf apply || printf dry-run)"
printf 'promote=%s\n' "$PROMOTE"
printf 'mask_backfill=%s\n' "$BACKFILL_MASK"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'snapshot_report=%s\n' "$SNAPSHOT_REPORT"
printf 'mask_report=%s\n' "$MASK_REPORT"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and --submit-host is empty"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_mode="ssh_bsub"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
    printf 'snapshot_report=%s\n' "$SNAPSHOT_REPORT"
    printf 'mask_report=%s\n' "$MASK_REPORT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
