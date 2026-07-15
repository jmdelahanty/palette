#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
OUTPUT_ROOT="${PALETTE_TABULAR_SNAPSHOT_ROOT:-/groups/johnson/johnsonlab/jeremy/recordings/logs/tabular_snapshot_bsub}"
RUN_ID=""
ZARR_PATH=""
FAMILY=""
SOURCE_RUN=""
OUTPUT_RUN=""
QUEUE="short"
MEM_GB=8
WALLTIME="0:30"
SHARD_ROWS=131072
APPLY=0
PROMOTE=1
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_tabular_snapshot_bsub.sh [options]

Render or submit one CPU job that plans or publishes an immutable indexed-
sharded tabular snapshot. The login host only submits bsub; all Zarr reads and
writes occur inside the LSF allocation. Default operation is a dry run.

Required:
  --run-id ID
  --zarr-path PATH
  --family FAMILY             keypoints_runs, refined_keypoints_runs, or refined_detect_runs
  --source-run NAME
  --output-run NAME

Options:
  --shard-rows N              Default: 131072
  --apply                     Write the snapshot; default is dry-run
  --no-promote                Complete without changing family pointers
  --palette-repo PATH
  --output-root PATH
  --submit-host HOST
  --queue NAME                Default: short
  --mem-gb N                  Default: 8
  --walltime H:MM             Default: 0:30
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
    --family) FAMILY="$2"; shift 2;;
    --source-run) SOURCE_RUN="$2"; shift 2;;
    --output-run) OUTPUT_RUN="$2"; shift 2;;
    --shard-rows) SHARD_ROWS="$2"; shift 2;;
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
case "$FAMILY" in
  keypoints_runs|refined_keypoints_runs|refined_detect_runs) ;;
  *) fail "unsupported --family: $FAMILY";;
esac
[[ -n "$SOURCE_RUN" && "$SOURCE_RUN" != */* ]] || fail "unsafe --source-run: $SOURCE_RUN"
[[ -n "$OUTPUT_RUN" && "$OUTPUT_RUN" != */* ]] || fail "unsafe --output-run: $OUTPUT_RUN"
[[ "$SHARD_ROWS" =~ ^[1-9][0-9]*$ ]] || fail "--shard-rows must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable: $PALETTE_REPO"
[[ -f "$PALETTE_REPO/src/fisheye/utils/publish_tabular_snapshot.py" ]] || \
  fail "snapshot publisher is missing from: $PALETTE_REPO"

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/report.json"
JOB_SCRIPT="$RUN_DIR/run_tabular_snapshot.sh"
STATUS_FILE="$RUN_DIR/status.txt"
SUBMISSION_FILE="$RUN_DIR/submission.txt"
EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_family="$(printf '%q' "$FAMILY")"
q_source="$(printf '%q' "$SOURCE_RUN")"
q_output="$(printf '%q' "$OUTPUT_RUN")"
q_shards="$(printf '%q' "$SHARD_ROWS")"
q_report="$(printf '%q' "$REPORT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
FAMILY=${q_family}
SOURCE_RUN=${q_source}
OUTPUT_RUN=${q_output}
SHARD_ROWS=${q_shards}
REPORT=${q_report}
STATUS_FILE=${q_status}
EXPECTED_COMMIT=${q_commit}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
cmd=(
  scripts/py -m fisheye.utils.publish_tabular_snapshot
  "\${ZARR_PATH}"
  --family "\${FAMILY}"
  --source-run "\${SOURCE_RUN}"
  --output-run "\${OUTPUT_RUN}"
  --shard-rows "\${SHARD_ROWS}"
  --json
)
JOBSCRIPT

if [[ "$APPLY" == "1" ]]; then
  printf 'cmd+=(--apply)\n' >>"$JOB_SCRIPT"
fi
if [[ "$PROMOTE" == "0" ]]; then
  printf 'cmd+=(--no-promote)\n' >>"$JOB_SCRIPT"
fi

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
"${cmd[@]}" >"${REPORT}"

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'report=%s\n' "${REPORT}"
} >"${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "tabular_snapshot_${RUN_ID}"
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
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'report=%s\n' "$REPORT"
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
    printf 'report=%s\n' "$REPORT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
