#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SOURCE_REPO="${PALETTE_SOURCE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
OUTPUT_ROOT="${PALETTE_YOLO_SHARD_MIGRATION_ROOT:-/groups/johnson/johnsonlab/jeremy/recordings/logs/immutable_yolo_sharding_migration_bsub}"
RUN_ID=""
QUEUE=""
MEM_GB=8
WALLTIME="4:00"
APPLY=0
SUBMIT=0
ZARR_PATHS=()
STAGES=()

usage() {
  cat <<'USAGE'
Usage: submit_immutable_yolo_sharding_migration_bsub.sh --run-id ID --zarr-path PATH... [options]

Render or submit one sequential LSF job that plans or applies the recoverable
selected-run YOLO sharding migration. The Citrus login host only submits bsub;
all Zarr reads, hashes, writes, publication, and validation run in the compute
allocation.

Required:
  --run-id ID                 Unique run identifier
  --zarr-path PATH            Analysis Zarr to migrate; repeatable

Options:
  --stage detect|keypoints|both
                              Repeatable; default: both
  --apply                     Apply after planning (default: dry-run)
  --palette-repo PATH         Cluster-visible Palette checkout
  --source-repo PATH          Workstation checkout supplying bundled module
  --output-root PATH          Shared logs/report root
  --submit-host HOST          Default: login1-citrus-poller
  --queue NAME
  --mem-gb N                  Default: 8
  --walltime H:MM             Default: 4:00
  --submit                    Submit through the Citrus poller
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
    --zarr-path) ZARR_PATHS+=("$2"); shift 2;;
    --stage) STAGES+=("$2"); shift 2;;
    --apply) APPLY=1; shift;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --source-repo) SOURCE_REPO="$2"; shift 2;;
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
[[ "${#ZARR_PATHS[@]}" -gt 0 ]] || fail "provide at least one --zarr-path"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ -n "$SUBMIT_HOST" ]] || fail "--submit-host must not be empty"
for stage in "${STAGES[@]}"; do
  [[ "$stage" == "detect" || "$stage" == "keypoints" || "$stage" == "both" ]] \
    || fail "invalid --stage: $stage"
done
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable: $PALETTE_REPO"
SOURCE_MODULE="$SOURCE_REPO/src/fisheye/utils/migrate_immutable_yolo_sharding.py"
[[ -f "$SOURCE_MODULE" ]] || fail "local migration module not found: $SOURCE_MODULE"
SOURCE_FILES=("$PALETTE_REPO/src/fisheye/shared/zarr_helpers.py")
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "$source_file" ]] || fail "migration dependency not found: $source_file"
done
for path in "${ZARR_PATHS[@]}"; do
  [[ -f "$path/zarr.json" ]] || fail "not a Zarr v3 root: $path"
done

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"
PATHS_FILE="$RUN_DIR/zarr_paths.txt"
REPORT="$RUN_DIR/report.json"
JOB_SCRIPT="$RUN_DIR/run_immutable_yolo_sharding_migration.sh"
STATUS_FILE="$RUN_DIR/status.txt"
SUBMISSION_FILE="$RUN_DIR/submission.txt"
BUNDLED_MODULE="$RUN_DIR/migrate_immutable_yolo_sharding.py"
printf '%s\n' "${ZARR_PATHS[@]}" >"$PATHS_FILE"
cp "$SOURCE_MODULE" "$BUNDLED_MODULE"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
EXPECTED_SOURCE_SHA256="$(sha256sum "$BUNDLED_MODULE" "${SOURCE_FILES[@]}" | awk '{print $1}' | sha256sum | awk '{print $1}')"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_paths="$(printf '%q' "$PATHS_FILE")"
q_report="$(printf '%q' "$REPORT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_module="$(printf '%q' "$BUNDLED_MODULE")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_source_sha="$(printf '%q' "$EXPECTED_SOURCE_SHA256")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
PATHS_FILE=${q_paths}
REPORT=${q_report}
STATUS_FILE=${q_status}
BUNDLED_MODULE=${q_module}
EXPECTED_COMMIT=${q_commit}
EXPECTED_SOURCE_SHA256=${q_source_sha}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
ACTUAL_SOURCE_SHA256="\$(sha256sum \
  "\${BUNDLED_MODULE}" \
  src/fisheye/shared/zarr_helpers.py | awk '{print \$1}' | sha256sum | awk '{print \$1}')"
if [[ "\${ACTUAL_SOURCE_SHA256}" != "\${EXPECTED_SOURCE_SHA256}" ]]; then
  printf 'Migration source hash mismatch: expected %s, found %s\n' \
    "\${EXPECTED_SOURCE_SHA256}" "\${ACTUAL_SOURCE_SHA256}" >&2
  exit 2
fi
mapfile -t ZARR_PATHS <"\${PATHS_FILE}"
cmd=(
  scripts/py "\${BUNDLED_MODULE}"
  --report-json "\${REPORT}"
)
JOBSCRIPT

if [[ "$APPLY" == "1" ]]; then
  printf 'cmd+=(--apply)\n' >>"$JOB_SCRIPT"
fi
for stage in "${STAGES[@]}"; do
  printf 'cmd+=(--stage %q)\n' "$stage" >>"$JOB_SCRIPT"
done

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
cmd+=("${ZARR_PATHS[@]}")
printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'source_sha256=%s\n' "${ACTUAL_SOURCE_SHA256}"
  printf 'report=%s\n' "${REPORT}"
} >"${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "yolo_shard_${RUN_ID}"
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
printf 'zarr_count=%s\n' "${#ZARR_PATHS[@]}"
printf 'stages=%s\n' "${STAGES[*]:-detect keypoints}"
printf 'submit_host=%s\n' "$SUBMIT_HOST"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'source_sha256=%s\n' "$EXPECTED_SOURCE_SHA256"
printf 'report=%s\n' "$REPORT"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
  submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=ssh_bsub\n'
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
    printf 'source_sha256=%s\n' "$EXPECTED_SOURCE_SHA256"
    printf 'report=%s\n' "$REPORT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
