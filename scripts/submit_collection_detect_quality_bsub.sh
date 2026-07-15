#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
OUTPUT_ROOT="${PALETTE_COLLECTION_DETECT_QUALITY_LOG_ROOT:-/groups/johnson/johnsonlab/jeremy/recordings/logs/collection_detect_quality_bsub}"
RUN_ID=""
ZARR_PATH=""
SOURCE_GROUP=""
OUTPUT_RUN=""
FRAME_COUNT=""
WIDTH=""
HEIGHT=""
EXPECTED_SUBJECT_COUNT=""
WORKERS=8
MEM_GB_PER_CORE=2
QUEUE="short"
WALLTIME="1:00"
APPLY=0
PROMOTE=1
SUBMIT=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: submit_collection_detect_quality_bsub.sh [options]

Render or submit one multicore LSF job that reads complete recording-level
detection shards, writes compact worker traces to node-local scratch, and then
publishes one validated indexed-sharded detect_quality_runs snapshot.

Required:
  --run-id ID
  --zarr-path PATH
  --source-group PATH
  --output-run NAME

Options:
  --recording-frame-count N
  --width N --height N
  --expected-subject-count N
  --workers N                         Default: 8
  --shard-rows N                      Default: 131072
  --row-chunk-rows N                  Default: 16384
  --frame-chunk-rows N                Default: 16384
  --jump-threshold N                  Default: 100
  --threshold-mode MODE               scaled|pixels|normalized
  --threshold-reference-width N       Default: 640
  --blip-gap-threshold N              Default: 10
  --relocation-confirm-count N        Default: 3
  --relocation-cluster-radius-fraction N
                                       Default: 0.5
  --apply                              Write and promote; default is dry-run
  --no-promote
  --palette-repo PATH
  --output-root PATH
  --submit-host HOST
  --queue NAME                         Default: short
  --mem-gb-per-core N                 Default: 2
  --walltime H:MM                     Default: 1:00
  --submit
  -h, --help
USAGE
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;;
    --zarr-path) ZARR_PATH="$2"; shift 2;;
    --source-group) SOURCE_GROUP="$2"; shift 2;;
    --output-run) OUTPUT_RUN="$2"; shift 2;;
    --recording-frame-count) FRAME_COUNT="$2"; EXTRA_ARGS+=(--recording-frame-count "$2"); shift 2;;
    --width) WIDTH="$2"; EXTRA_ARGS+=(--width "$2"); shift 2;;
    --height) HEIGHT="$2"; EXTRA_ARGS+=(--height "$2"); shift 2;;
    --expected-subject-count) EXPECTED_SUBJECT_COUNT="$2"; EXTRA_ARGS+=(--expected-subject-count "$2"); shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --shard-rows|--row-chunk-rows|--frame-chunk-rows|--jump-threshold|--threshold-mode|--threshold-reference-width|--blip-gap-threshold|--relocation-confirm-count|--relocation-cluster-radius-fraction)
      EXTRA_ARGS+=("$1" "$2"); shift 2;;
    --apply) APPLY=1; shift;;
    --no-promote) PROMOTE=0; shift;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb-per-core) MEM_GB_PER_CORE="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$RUN_ID" ]] || fail "--run-id is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --run-id: $RUN_ID"
[[ -n "$ZARR_PATH" && -f "$ZARR_PATH/zarr.json" ]] || fail "--zarr-path must be a Zarr v3 root"
[[ -n "$SOURCE_GROUP" ]] || fail "--source-group is required"
[[ -n "$OUTPUT_RUN" && "$OUTPUT_RUN" != */* ]] || fail "--output-run must be a safe group name"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "--workers must be positive"
[[ "$MEM_GB_PER_CORE" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb-per-core must be positive"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette checkout is unavailable: $PALETTE_REPO"

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"
JOB_SCRIPT="$RUN_DIR/run_collection_detect_quality.sh"
REPORT="$RUN_DIR/report.json"
STATUS="$RUN_DIR/status.txt"
EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"

shell_join() { printf '%q ' "$@"; }
CMD=(
  scripts/py -m fisheye.refinement.detect_quality_collection
  "$ZARR_PATH"
  --source-group "$SOURCE_GROUP"
  --output-run "$OUTPUT_RUN"
  --workers "$WORKERS"
  --json
  "${EXTRA_ARGS[@]}"
)
[[ "$APPLY" == "1" ]] && CMD+=(--apply)
[[ "$PROMOTE" == "0" ]] && CMD+=(--no-promote)
CMD_SHELL="$(shell_join "${CMD[@]}")"

cat >"$JOB_SCRIPT" <<JOB
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=$(printf '%q' "$PALETTE_REPO")
EXPECTED_COMMIT=$(printf '%q' "$EXPECTED_COMMIT")
REPORT=$(printf '%q' "$REPORT")
STATUS=$(printf '%q' "$STATUS")
cd "\$PALETTE_REPO"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\$ACTUAL_COMMIT" == "\$EXPECTED_COMMIT" ]] || {
  printf 'Palette commit mismatch: expected %s, found %s\n' "\$EXPECTED_COMMIT" "\$ACTUAL_COMMIT" >&2
  exit 2
}
scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\$scratch_user" && -w "/scratch/\$scratch_user" ]]; then
  SCRATCH_ROOT="/scratch/\$scratch_user/\${LSB_JOBID}/collection_detect_quality"
else
  SCRATCH_ROOT="\${TMPDIR:-/tmp}/palette_collection_detect_quality_\${LSB_JOBID:-manual}"
fi
mkdir -p "\$SCRATCH_ROOT"
trap 'rm -rf "\$SCRATCH_ROOT"' EXIT
cmd=(${CMD_SHELL})
cmd+=(--work-dir "\$SCRATCH_ROOT")
printf 'command='; printf '%q ' "\${cmd[@]}"; printf '\n'
"\${cmd[@]}" >"\$REPORT"
{
  printf 'status=complete\n'
  printf 'operation=%s\n' $(printf '%q' "$([[ "$APPLY" == "1" ]] && printf apply || printf dry-run)")
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "\$ACTUAL_COMMIT"
  printf 'report=%s\n' "\$REPORT"
} >"\$STATUS"
JOB
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "collection_detect_quality_${RUN_ID}"
  -n "$WORKERS"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB_PER_CORE}G] span[hosts=1]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
  -q "$QUEUE"
)
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'operation=%s\n' "$([[ "$APPLY" == "1" ]] && printf apply || printf dry-run)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'report=%s\n' "$REPORT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  submit_output="$(ssh "$SUBMIT_HOST" "$(shell_join "${BSUB_COMMAND[@]}")")"
  printf '%s\n' "$submit_output"
  job_id="$(sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' <<<"$submit_output" | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse submitted job id"
  printf 'job_id=%s\n' "$job_id"
fi
