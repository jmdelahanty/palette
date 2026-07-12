#!/usr/bin/env bash
set -euo pipefail
umask 0002

COLLECTION_MANIFEST=""
EXPORT_RUN_ID=""
STATS_RUN_ID=""
OUTPUT_ROOT="${PALETTE_ANALYTICS_EXPORT_ROOT:-/groups/johnson/johnsonlab/palette_analytics}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
LOG_DIR=""
QUEUE="normal"
NCORES=4
MEM_GB=16
WALLTIME="2:00"
RUN_STATISTICS=1
INDEX_REGISTRY=0
SUBMIT=0
TABLES="chaser_epoch_spatial_occupancy_zones,chaser_epoch_distance_summary,chaser_epoch_behavior_summary,chaser_epoch_bout_events,chaser_epoch_bout_histogram,chaser_epoch_inter_bout_interval_histogram,chaser_epoch_center_distance_histogram,chaser_speed_distance_bins,chaser_epoch_distance_histogram,chaser_cra_primary_endpoint_summary,chaser_cra_primary_endpoint_object_phase,chaser_cra_quadrant_occupancy,chaser_cra_near_field_summary,chaser_cra_near_field_object_phase,chaser_cra_near_field_radial_density,chaser_cra_near_field_distance_cdf,chaser_egocentric_epoch_summary,chaser_egocentric_distance_bearing_histogram"

usage() {
  cat <<'USAGE'
Usage: submit_analytics_export_bsub.sh --collection-manifest PATH --export-run-id ID [options]

Create one fail-closed LSF job that exports a virtual collection directly to
the shared Palette analytics root, validates every Parquet part, computes linked
group statistics, and validates the statistics export.

Required:
  --collection-manifest PATH   Virtual collection manifest on shared storage
  --export-run-id ID           Immutable base-export run ID

Options:
  --stats-run-id ID            Linked statistics run ID (default: <export-id>_stats)
  --output-root PATH           Shared analytics root
                               (default: /groups/johnson/johnsonlab/palette_analytics)
  --palette-repo PATH          Cluster-visible Palette checkout
                               (default: /groups/.../jeremy/gitrepos/palette)
  --tables CSV                 V2 tables (default: all chaser tables)
  --skip-statistics            Do not compute the linked chaser statistics export
  --index-registry             Index the completed base export in the shared registry
  --registry PATH              Registry path used with --index-registry
  --queue NAME                 LSF queue (default: normal)
  --ncores N                   CPU slots and exporter workers (default: 4)
  --mem-gb N                   Memory request in GB (default: 16)
  --walltime H:MM              LSF wall time (default: 2:00)
  --log-dir PATH               Submission artifacts (default: <output-root>/logs/lsf)
  --submit                     Submit with bsub; without this flag, only render the job
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
    --export-run-id) EXPORT_RUN_ID="$2"; shift 2;;
    --stats-run-id) STATS_RUN_ID="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --tables) TABLES="$2"; shift 2;;
    --skip-statistics) RUN_STATISTICS=0; shift;;
    --index-registry) INDEX_REGISTRY=1; shift;;
    --registry) REGISTRY="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "Unknown argument: $1";;
  esac
done

[[ -n "$COLLECTION_MANIFEST" ]] || fail "--collection-manifest is required"
[[ -n "$EXPORT_RUN_ID" ]] || fail "--export-run-id is required"
[[ "$EXPORT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Unsafe --export-run-id: $EXPORT_RUN_ID"
if [[ -z "$STATS_RUN_ID" ]]; then STATS_RUN_ID="${EXPORT_RUN_ID}_stats"; fi
[[ "$STATS_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Unsafe --stats-run-id: $STATS_RUN_ID"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be a positive integer"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be a positive integer"
[[ -f "$COLLECTION_MANIFEST" ]] || fail "Collection manifest not found: $COLLECTION_MANIFEST"
[[ -d "$PALETTE_REPO/.git" ]] || fail "Palette checkout not found: $PALETTE_REPO"
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

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_collection="$(printf '%q' "$COLLECTION_MANIFEST")"
q_output="$(printf '%q' "$OUTPUT_ROOT")"
q_export_id="$(printf '%q' "$EXPORT_RUN_ID")"
q_stats_id="$(printf '%q' "$STATS_RUN_ID")"
q_tables="$(printf '%q' "$TABLES")"
q_registry="$(printf '%q' "$REGISTRY")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_validation="$(printf '%q' "$VALIDATION_JSON")"
q_status="$(printf '%q' "$STATUS_FILE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
COLLECTION_MANIFEST=${q_collection}
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

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
mkdir -p "\${OUTPUT_ROOT}" "\$(dirname -- "\${VALIDATION_JSON}")"
export MPLCONFIGDIR="\$(dirname -- "\${VALIDATION_JSON}")/matplotlib"

export_cmd=(
  scripts/py -m fisheye.utils.export_cross_recording_analytics
  --collection-manifest "\${COLLECTION_MANIFEST}"
  --output-root "\${OUTPUT_ROOT}"
  --tables "\${TABLES}"
  --jobs "\${NCORES}"
  --export-run-id "\${EXPORT_RUN_ID}"
)
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
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'collection_manifest=%s\n' "$COLLECTION_MANIFEST"
printf 'output_root=%s\n' "$OUTPUT_ROOT"
printf 'export_run_id=%s\n' "$EXPORT_RUN_ID"
printf 'stats_run_id=%s\n' "$STATS_RUN_ID"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  command -v bsub >/dev/null 2>&1 || fail "bsub is not available on this host"
  "${BSUB_COMMAND[@]}"
fi
