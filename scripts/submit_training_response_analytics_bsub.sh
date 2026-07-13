#!/usr/bin/env bash
set -euo pipefail
umask 0002

SOURCE_EXPORT_ROOT="${PALETTE_ANALYTICS_EXPORT_ROOT:-/groups/johnson/johnsonlab/palette_analytics}"
SOURCE_EXPORT_RUN_ID=""
OUTPUT_ROOT="${PALETTE_TRAINING_RESPONSE_ANALYTICS_ROOT:-/groups/johnson/johnsonlab/palette_training_response_analytics}"
ANALYSIS_RUN_ID=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
QUEUE=""
NCORES=1
MEM_GB=16
WALLTIME="1:00"
MIN_VALID_POSITION_FRACTION=0.75
RELATIVE_SCORE_THRESHOLD=0.75
CLUSTER_MAX_COMPONENTS=6
CLUSTER_MIN_ROWS_PER_COMPONENT=10
CLUSTER_STABILITY_THRESHOLD=0.60
CLUSTER_STABILITY_RESAMPLES=25
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_training_response_analytics_bsub.sh \
  --source-export-run-id ID --analysis-run-id ID [options]

Render or submit one fail-closed CPU LSF job that reads an immutable Palette
analytics export and publishes separate whole-training response analytics.
No Parquet analysis is performed on the login node.

Required:
  --source-export-run-id ID
  --analysis-run-id ID

Options:
  --source-export-root PATH
  --output-root PATH
  --palette-repo PATH
  --submit-host HOST
  --min-valid-position-fraction VALUE
  --relative-score-threshold VALUE
  --cluster-max-components N
  --cluster-min-rows-per-component N
  --cluster-stability-threshold VALUE
  --cluster-stability-resamples N
  --queue NAME
  --mem-gb N                  Memory request (default: 16)
  --walltime H:MM             Wall time (default: 1:00)
  --log-dir PATH              Default: <output-root>/logs/lsf
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
    --source-export-root) SOURCE_EXPORT_ROOT="$2"; shift 2;;
    --source-export-run-id) SOURCE_EXPORT_RUN_ID="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --analysis-run-id) ANALYSIS_RUN_ID="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --min-valid-position-fraction) MIN_VALID_POSITION_FRACTION="$2"; shift 2;;
    --relative-score-threshold) RELATIVE_SCORE_THRESHOLD="$2"; shift 2;;
    --cluster-max-components) CLUSTER_MAX_COMPONENTS="$2"; shift 2;;
    --cluster-min-rows-per-component) CLUSTER_MIN_ROWS_PER_COMPONENT="$2"; shift 2;;
    --cluster-stability-threshold) CLUSTER_STABILITY_THRESHOLD="$2"; shift 2;;
    --cluster-stability-resamples) CLUSTER_STABILITY_RESAMPLES="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$SOURCE_EXPORT_RUN_ID" ]] || fail "--source-export-run-id is required"
[[ -n "$ANALYSIS_RUN_ID" ]] || fail "--analysis-run-id is required"
[[ "$SOURCE_EXPORT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
  fail "unsafe --source-export-run-id: $SOURCE_EXPORT_RUN_ID"
[[ "$ANALYSIS_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
  fail "unsafe --analysis-run-id: $ANALYSIS_RUN_ID"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be a positive integer"
[[ "$CLUSTER_MAX_COMPONENTS" =~ ^[1-9][0-9]*$ ]] || \
  fail "--cluster-max-components must be positive"
[[ "$CLUSTER_MIN_ROWS_PER_COMPONENT" =~ ^[1-9][0-9]*$ ]] || \
  fail "--cluster-min-rows-per-component must be positive"
(( CLUSTER_MIN_ROWS_PER_COMPONENT >= 2 )) || \
  fail "--cluster-min-rows-per-component must be at least 2"
[[ "$CLUSTER_STABILITY_RESAMPLES" =~ ^[0-9]+$ ]] || \
  fail "--cluster-stability-resamples must be non-negative"
SOURCE_MANIFEST="${SOURCE_EXPORT_ROOT}/v1/manifests/export_run_id=${SOURCE_EXPORT_RUN_ID}.json"
[[ -f "$SOURCE_MANIFEST" ]] || fail "source manifest not found: $SOURCE_MANIFEST"
[[ -d "$PALETTE_REPO/.git" ]] || fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ ! -e "${OUTPUT_ROOT}/v1/manifests/analysis_run_id=${ANALYSIS_RUN_ID}.json" ]] || \
  fail "analysis manifest already exists: $ANALYSIS_RUN_ID"

if [[ -z "$LOG_DIR" ]]; then LOG_DIR="${OUTPUT_ROOT}/logs/lsf"; fi
RUN_DIR="${LOG_DIR}/training_response_${ANALYSIS_RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_training_response.sh"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_source_root="$(printf '%q' "$SOURCE_EXPORT_ROOT")"
q_source_run="$(printf '%q' "$SOURCE_EXPORT_RUN_ID")"
q_output_root="$(printf '%q' "$OUTPUT_ROOT")"
q_analysis_run="$(printf '%q' "$ANALYSIS_RUN_ID")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_valid="$(printf '%q' "$MIN_VALID_POSITION_FRACTION")"
q_threshold="$(printf '%q' "$RELATIVE_SCORE_THRESHOLD")"
q_components="$(printf '%q' "$CLUSTER_MAX_COMPONENTS")"
q_min_rows="$(printf '%q' "$CLUSTER_MIN_ROWS_PER_COMPONENT")"
q_stability="$(printf '%q' "$CLUSTER_STABILITY_THRESHOLD")"
q_resamples="$(printf '%q' "$CLUSTER_STABILITY_RESAMPLES")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
SOURCE_EXPORT_ROOT=${q_source_root}
SOURCE_EXPORT_RUN_ID=${q_source_run}
OUTPUT_ROOT=${q_output_root}
ANALYSIS_RUN_ID=${q_analysis_run}
EXPECTED_COMMIT=${q_commit}
STATUS_FILE=${q_status}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi

scripts/py -m fisheye.training_response.workflow \
  --source-export-root "\${SOURCE_EXPORT_ROOT}" \
  --source-export-run-id "\${SOURCE_EXPORT_RUN_ID}" \
  --output-root "\${OUTPUT_ROOT}" \
  --analysis-run-id "\${ANALYSIS_RUN_ID}" \
  --min-valid-position-fraction ${q_valid} \
  --relative-score-threshold ${q_threshold} \
  --cluster-max-components ${q_components} \
  --cluster-min-rows-per-component ${q_min_rows} \
  --cluster-stability-threshold ${q_stability} \
  --cluster-stability-resamples ${q_resamples}

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'source_export_run_id=%s\n' "\${SOURCE_EXPORT_RUN_ID}"
  printf 'analysis_run_id=%s\n' "\${ANALYSIS_RUN_ID}"
} >"\${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "training_response_${ANALYSIS_RUN_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'source_export_run_id=%s\n' "$SOURCE_EXPORT_RUN_ID"
printf 'analysis_run_id=%s\n' "$ANALYSIS_RUN_ID"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and --submit-host is empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
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
  } >"$SUBMISSION_FILE"
fi
