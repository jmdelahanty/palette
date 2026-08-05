#!/usr/bin/env bash
set -euo pipefail
umask 0002

COLLECTION_MANIFEST=""
CHASER_AUTHORITY_MANIFEST=""
CHASER_AUTHORITY_SHA256=""
EXPORT_RUN_ID=""
STATS_RUN_ID=""
OUTPUT_ROOT="${PALETTE_ANALYTICS_EXPORT_ROOT:-/groups/johnson/johnsonlab/palette_analytics}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
LOG_DIR=""
QUEUE=""
NCORES=4
MEM_GB=16
WALLTIME="2:00"
RUN_STATISTICS=1
INDEX_REGISTRY=0
SUBMIT=0
DEPENDENCY_DONE=()
BASELINE_TIME_BIN_S=5
BASELINE_SAMPLE_RATE_HZ=10
BASELINE_FULL_RESOLUTION_SAMPLES=0
BASELINE_SPATIAL_GRID_SIZE=12
INCLUDE_BASELINE_SAMPLES=0
TABLES="baseline_behavior_summary,baseline_behavior_time_bins,position_occupancy_histogram_2d,chaser_epoch_spatial_occupancy_zones,chaser_epoch_distance_summary,chaser_epoch_behavior_summary,chaser_epoch_bout_events,chaser_epoch_bout_histogram,chaser_epoch_inter_bout_interval_histogram,chaser_epoch_center_distance_histogram,chaser_speed_distance_bins,chaser_epoch_distance_histogram,chaser_quadrant_occupancy_summary,chaser_quadrant_occupancy_chaser_phase,chaser_quadrant_occupancy_density,chaser_near_field_occupancy_summary,chaser_near_field_occupancy_chaser_phase,chaser_near_field_occupancy_radial_density,chaser_near_field_occupancy_distance_cdf,chaser_egocentric_epoch_summary,chaser_egocentric_distance_bearing_histogram"

usage() {
  cat <<'USAGE'
Usage: submit_analytics_export_bsub.sh --collection-manifest PATH --export-run-id ID [options]

Create one fail-closed LSF job that exports a virtual collection directly to
the shared Palette analytics root, validates every Parquet part, computes linked
group statistics, and validates the statistics export.

Required:
  --collection-manifest PATH   Virtual collection manifest on shared storage
  --chaser-authority-manifest PATH
                               Exact chaser export authority-set JSON
  --export-run-id ID           Immutable base-export run ID

Options:
  --stats-run-id ID            Linked statistics run ID (default: <export-id>_stats)
  --output-root PATH           Shared analytics root
                               (default: /groups/johnson/johnsonlab/palette_analytics)
  --palette-repo PATH          Cluster-visible Palette checkout
                               (default: /groups/.../jeremy/gitrepos/palette)
  --submit-host HOST           SSH host used when bsub is unavailable locally
                               (default: login1-citrus-poller)
  --tables CSV                 V2 tables (default: all chaser tables)
  --chaser-authority-sha256 HEX
                               Optional expected authority-manifest file digest
  --baseline-time-bin-s S      Baseline behavior bin width (default: 5)
  --include-baseline-samples   Add optional baseline_kinematic_samples table
  --baseline-sample-rate-hz HZ Requested baseline sample rate (default: 10)
  --baseline-full-resolution-samples
                               Export every baseline source sample; also enables table
  --baseline-spatial-grid-size N
                               Per-axis entropy grid size (default: 12)
  --skip-statistics            Do not compute the linked chaser statistics export
  --index-registry             Index the completed base export in the shared registry
  --registry PATH              Registry path used with --index-registry
  --queue NAME                 LSF queue (default: cluster default)
  --ncores N                   CPU slots and exporter workers (default: 4)
  --mem-gb N                   Memory request in GB (default: 16)
  --walltime H:MM              LSF wall time (default: 2:00)
  --log-dir PATH               Submission artifacts (default: <output-root>/logs/lsf)
  --dependency-done JOBID      Submit after this LSF job succeeds. May repeat.
  --submit                     Submit with local bsub or SSH only the bsub command
                               to --submit-host; otherwise only render the job
  -h, --help                   Show this message

The job refuses existing run manifests and validates the exact Palette commit
captured at submission time. Export manifests are written last by the exporter,
so an interrupted run is not discoverable as a completed dataset.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection-manifest) COLLECTION_MANIFEST="$2"; shift 2;;
    --chaser-authority-manifest) CHASER_AUTHORITY_MANIFEST="$2"; shift 2;;
    --chaser-authority-sha256) CHASER_AUTHORITY_SHA256="$2"; shift 2;;
    --export-run-id) EXPORT_RUN_ID="$2"; shift 2;;
    --stats-run-id) STATS_RUN_ID="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --tables) TABLES="$2"; shift 2;;
    --baseline-time-bin-s) BASELINE_TIME_BIN_S="$2"; shift 2;;
    --include-baseline-samples) INCLUDE_BASELINE_SAMPLES=1; shift;;
    --baseline-sample-rate-hz) BASELINE_SAMPLE_RATE_HZ="$2"; shift 2;;
    --baseline-full-resolution-samples)
      BASELINE_FULL_RESOLUTION_SAMPLES=1; INCLUDE_BASELINE_SAMPLES=1; shift;;
    --baseline-spatial-grid-size) BASELINE_SPATIAL_GRID_SIZE="$2"; shift 2;;
    --skip-statistics) RUN_STATISTICS=0; shift;;
    --index-registry) INDEX_REGISTRY=1; shift;;
    --registry) REGISTRY="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --dependency-done) DEPENDENCY_DONE+=("$2"); shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "Unknown argument: $1";;
  esac
done

[[ -n "$COLLECTION_MANIFEST" ]] || fail "--collection-manifest is required"
[[ -n "$CHASER_AUTHORITY_MANIFEST" ]] || \
  fail "--chaser-authority-manifest is required"
[[ -n "$EXPORT_RUN_ID" ]] || fail "--export-run-id is required"
[[ "$EXPORT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Unsafe --export-run-id: $EXPORT_RUN_ID"
if [[ -z "$STATS_RUN_ID" ]]; then STATS_RUN_ID="${EXPORT_RUN_ID}_stats"; fi
[[ "$STATS_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Unsafe --stats-run-id: $STATS_RUN_ID"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be a positive integer"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be a positive integer"
[[ "$BASELINE_SPATIAL_GRID_SIZE" =~ ^[0-9]+$ ]] || \
  fail "--baseline-spatial-grid-size must be an integer >= 2"
(( BASELINE_SPATIAL_GRID_SIZE >= 2 )) || \
  fail "--baseline-spatial-grid-size must be an integer >= 2"
if [[ "$INCLUDE_BASELINE_SAMPLES" == "1" && ",$TABLES," != *,baseline_kinematic_samples,* ]]; then
  TABLES="${TABLES},baseline_kinematic_samples"
fi
for dependency_job_id in "${DEPENDENCY_DONE[@]}"; do
  [[ "$dependency_job_id" =~ ^[1-9][0-9]*$ ]] || \
    fail "--dependency-done must be a positive numeric LSF job ID: $dependency_job_id"
done
if [[ "${#DEPENDENCY_DONE[@]}" == "0" ]]; then
  [[ -f "$COLLECTION_MANIFEST" ]] || fail "Collection manifest not found: $COLLECTION_MANIFEST"
  [[ -f "$CHASER_AUTHORITY_MANIFEST" ]] || \
    fail "Chaser authority manifest not found: $CHASER_AUTHORITY_MANIFEST"
  if [[ -z "$CHASER_AUTHORITY_SHA256" ]]; then
    CHASER_AUTHORITY_SHA256="$(sha256sum "$CHASER_AUTHORITY_MANIFEST" | awk '{print $1}')"
  fi
fi
if [[ -n "$CHASER_AUTHORITY_SHA256" && ! "$CHASER_AUTHORITY_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  fail "--chaser-authority-sha256 must be one lowercase SHA-256"
fi
git -C "$PALETTE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable: $PALETTE_REPO"
if [[ "$INDEX_REGISTRY" == "1" ]]; then
  [[ -n "$REGISTRY" ]] || fail "--index-registry requires --registry"
fi

if [[ -z "$LOG_DIR" ]]; then LOG_DIR="${OUTPUT_ROOT}/logs/lsf"; fi
RUN_DIR="${LOG_DIR}/analytics_export_${EXPORT_RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "Submission run directory already exists: $RUN_DIR"
[[ ! -e "${OUTPUT_ROOT}/v1/manifests/export_run_id=${EXPORT_RUN_ID}.json" ]] || \
  fail "Base export manifest already exists: $EXPORT_RUN_ID"
if [[ "$RUN_STATISTICS" == "1" ]]; then
  [[ ! -e "${OUTPUT_ROOT}/v1/manifests/export_run_id=${STATS_RUN_ID}.json" ]] || \
    fail "Statistics export manifest already exists: $STATS_RUN_ID"
fi

mkdir -p "$RUN_DIR"
EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_analytics_export.sh"
VALIDATION_JSON="${RUN_DIR}/validation.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_collection="$(printf '%q' "$COLLECTION_MANIFEST")"
q_chaser_authority="$(printf '%q' "$CHASER_AUTHORITY_MANIFEST")"
q_chaser_authority_sha256="$(printf '%q' "$CHASER_AUTHORITY_SHA256")"
q_output="$(printf '%q' "$OUTPUT_ROOT")"
q_export_id="$(printf '%q' "$EXPORT_RUN_ID")"
q_stats_id="$(printf '%q' "$STATS_RUN_ID")"
q_tables="$(printf '%q' "$TABLES")"
q_registry="$(printf '%q' "$REGISTRY")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_validation="$(printf '%q' "$VALIDATION_JSON")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_baseline_time_bin_s="$(printf '%q' "$BASELINE_TIME_BIN_S")"
q_baseline_sample_rate_hz="$(printf '%q' "$BASELINE_SAMPLE_RATE_HZ")"
q_baseline_spatial_grid_size="$(printf '%q' "$BASELINE_SPATIAL_GRID_SIZE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
COLLECTION_MANIFEST=${q_collection}
CHASER_AUTHORITY_MANIFEST=${q_chaser_authority}
CHASER_AUTHORITY_SHA256=${q_chaser_authority_sha256}
OUTPUT_ROOT=${q_output}
EXPORT_RUN_ID=${q_export_id}
STATS_RUN_ID=${q_stats_id}
TABLES=${q_tables}
REGISTRY=${q_registry}
EXPECTED_COMMIT=${q_commit}
VALIDATION_JSON=${q_validation}
STATUS_FILE=${q_status}
NCORES=${NCORES}
RUN_STATISTICS=${RUN_STATISTICS}
INDEX_REGISTRY=${INDEX_REGISTRY}
BASELINE_TIME_BIN_S=${q_baseline_time_bin_s}
BASELINE_SAMPLE_RATE_HZ=${q_baseline_sample_rate_hz}
BASELINE_FULL_RESOLUTION_SAMPLES=${BASELINE_FULL_RESOLUTION_SAMPLES}
BASELINE_SPATIAL_GRID_SIZE=${q_baseline_spatial_grid_size}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ ! -f "\${COLLECTION_MANIFEST}" ]]; then
  printf 'Collection manifest is unavailable after dependencies completed: %s\n' \
    "\${COLLECTION_MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "\${CHASER_AUTHORITY_MANIFEST}" ]]; then
  printf 'Chaser authority manifest is unavailable after dependencies completed: %s\n' \
    "\${CHASER_AUTHORITY_MANIFEST}" >&2
  exit 2
fi
mkdir -p "\${OUTPUT_ROOT}" "\$(dirname -- "\${VALIDATION_JSON}")"
export MPLCONFIGDIR="\$(dirname -- "\${VALIDATION_JSON}")/matplotlib"

export_cmd=(
  scripts/py -m fisheye.utils.export_cross_recording_analytics
  --collection-manifest "\${COLLECTION_MANIFEST}"
  --chaser-authority-manifest "\${CHASER_AUTHORITY_MANIFEST}"
  --output-root "\${OUTPUT_ROOT}"
  --tables "\${TABLES}"
  --jobs "\${NCORES}"
  --export-run-id "\${EXPORT_RUN_ID}"
  --baseline-time-bin-s "\${BASELINE_TIME_BIN_S}"
  --baseline-sample-rate-hz "\${BASELINE_SAMPLE_RATE_HZ}"
  --baseline-spatial-grid-size "\${BASELINE_SPATIAL_GRID_SIZE}"
)
if [[ -n "\${CHASER_AUTHORITY_SHA256}" ]]; then
  export_cmd+=(--chaser-authority-sha256 "\${CHASER_AUTHORITY_SHA256}")
fi
if [[ "\${BASELINE_FULL_RESOLUTION_SAMPLES}" == "1" ]]; then
  export_cmd+=(--baseline-full-resolution-samples)
fi
if [[ "\${INDEX_REGISTRY}" == "1" ]]; then
  export_cmd+=(--index-registry --registry "\${REGISTRY}")
fi
printf 'export_command='; printf '%q ' "\${export_cmd[@]}"; printf '\n'
"\${export_cmd[@]}"

scripts/py -m fisheye.utils.validate_analytics_export \
  --export-root "\${OUTPUT_ROOT}" \
  --export-run-id "\${EXPORT_RUN_ID}" \
  --json-output "\${VALIDATION_JSON}"

if [[ "\${RUN_STATISTICS}" == "1" ]]; then
  scripts/py -m fisheye.utils.compute_group_statistics \
    --profile chaser \
    --source-export-run-id "\${EXPORT_RUN_ID}" \
    --export-root "\${OUTPUT_ROOT}" \
    --stats-run-id "\${STATS_RUN_ID}" \
    --apply
  scripts/py -m fisheye.utils.validate_analytics_export \
    --export-root "\${OUTPUT_ROOT}" \
    --export-run-id "\${STATS_RUN_ID}"
fi

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'export_root=%s\n' "\${OUTPUT_ROOT}"
  printf 'export_run_id=%s\n' "\${EXPORT_RUN_ID}"
  printf 'stats_run_id=%s\n' "\${STATS_RUN_ID}"
} >"\${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "analytics_export_${EXPORT_RUN_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ "${#DEPENDENCY_DONE[@]}" -gt 0 ]]; then
  dependency_expression=""
  for dependency_job_id in "${DEPENDENCY_DONE[@]}"; do
    if [[ -n "$dependency_expression" ]]; then dependency_expression+=" && "; fi
    dependency_expression+="done(${dependency_job_id})"
  done
  BSUB_ARGS+=(-w "$dependency_expression")
fi
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'collection_manifest=%s\n' "$COLLECTION_MANIFEST"
printf 'output_root=%s\n' "$OUTPUT_ROOT"
printf 'export_run_id=%s\n' "$EXPORT_RUN_ID"
printf 'stats_run_id=%s\n' "$STATS_RUN_ID"
printf 'submit_host=%s\n' "$SUBMIT_HOST"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  submit_mode=""
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub is unavailable locally and --submit-host is empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "Could not parse an LSF job ID from submission output"
  submission_tmp="${SUBMISSION_FILE}.tmp.$$"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'job_script=%s\n' "$JOB_SCRIPT"
    printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
    printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$submission_tmp"
  mv "$submission_tmp" "$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
  printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
  printf 'status_file=%s\n' "$STATUS_FILE"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
