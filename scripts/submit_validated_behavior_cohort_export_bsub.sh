#!/usr/bin/env bash
set -euo pipefail
umask 0002

PLAN=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
QUEUE=""
MAX_ACTIVE=12
MEM_GB=8
WALLTIME="1:00"
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_validated_behavior_cohort_export_bsub.sh --plan PATH [options]

Render or submit one bounded recording-shard array followed by a serialized
manifest-last finalizer for an existing validated-behavior export plan.

Required:
  --plan PATH          Exact immutable export-plan JSON.

Options:
  --palette-repo PATH  Commit-pinned cluster deployment used by the plan.
  --log-dir PATH       New submission directory (default: beside shard root).
  --submit-host HOST   SSH host used when bsub is unavailable locally.
  --queue NAME         LSF queue (default: cluster default).
  --max-active N       Maximum simultaneously active shard jobs (default: 12).
  --mem-gb N           Memory per shard/finalizer job (default: 8).
  --walltime H:MM      Wall time per job (default: 1:00).
  --submit             Submit the array and dependent finalizer.
  -h, --help           Show this help.

The plan must bind this exact clean Palette commit and absolute deployment
path. Workers never mutate source Zarrs or the registry. The finalizer runs
only after every array element succeeds and commits the export manifest last.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "Unknown argument: $1";;
  esac
done

[[ -n "$PLAN" ]] || fail "--plan is required"
[[ -f "$PLAN" && -r "$PLAN" ]] || fail "Plan is not a readable file: $PLAN"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is unavailable: $PALETTE_REPO"
[[ "$MAX_ACTIVE" =~ ^[1-9][0-9]*$ ]] || fail "--max-active must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"

PLAN="$(realpath -- "$PLAN")"
PALETTE_REPO="$(realpath -- "$PALETTE_REPO")"
plan_info="$("$PALETTE_REPO/scripts/py" -c '
import sys
from fisheye.analytics_exports.validated_behavior_cohort import read_validated_behavior_export_plan
from fisheye.analytics_exports.validated_behavior_profiles import profile_id_from_record, resolve_validated_behavior_profile
profile = resolve_validated_behavior_profile(profile_id_from_record(sys.argv[1], record_kind="export plan"))
p, _, _ = read_validated_behavior_export_plan(sys.argv[1], table_specs=profile.table_specs)
print(p["member_count"])
print(p["export_run_id"])
print(p["software_authority"]["commit"])
print(p["software_authority"]["deployment_path"])
print(p["shard_root"])
' "$PLAN")"
mapfile -t plan_fields <<<"$plan_info"
[[ "${#plan_fields[@]}" == "5" ]] || fail "Could not read the exact plan identity"
MEMBER_COUNT="${plan_fields[0]}"
EXPORT_RUN_ID="${plan_fields[1]}"
EXPECTED_COMMIT="${plan_fields[2]}"
EXPECTED_REPO="${plan_fields[3]}"
SHARD_ROOT="${plan_fields[4]}"
[[ "$MEMBER_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "Plan member count is invalid"
[[ "$EXPORT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Plan export run ID is unsafe"
[[ "$EXPECTED_REPO" == "$PALETTE_REPO" ]] || \
  fail "--palette-repo differs from the plan deployment path"
ACTUAL_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || fail "Palette commit differs from plan"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette deployment worktree is not clean"

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${SHARD_ROOT}/submission_logs"
fi
RUN_DIR="${LOG_DIR}/validated_behavior_export_${EXPORT_RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "Submission run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

SHARD_SCRIPT="${RUN_DIR}/run_shard.sh"
FINALIZE_SCRIPT="${RUN_DIR}/run_finalize.sh"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_plan="$(printf '%q' "$PLAN")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"

cat >"$SHARD_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
PLAN=${q_plan}
EXPECTED_COMMIT=${q_commit}
cd "\${PALETTE_REPO}"
[[ "\$(git rev-parse HEAD)" == "\${EXPECTED_COMMIT}" ]] || {
  printf 'Palette commit mismatch\n' >&2
  exit 2
}
scripts/py -m fisheye.utils.materialize_validated_behavior_cohort_export shard \
  --plan "\${PLAN}" \
  --member-ordinal "\${LSB_JOBINDEX}"
JOBSCRIPT

cat >"$FINALIZE_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
PLAN=${q_plan}
EXPECTED_COMMIT=${q_commit}
STATUS_FILE=${q_status}
cd "\${PALETTE_REPO}"
[[ "\$(git rev-parse HEAD)" == "\${EXPECTED_COMMIT}" ]] || {
  printf 'Palette commit mismatch\n' >&2
  exit 2
}
scripts/py -m fisheye.utils.materialize_validated_behavior_cohort_export finalize \
  --plan "\${PLAN}"
scripts/py -m fisheye.utils.materialize_validated_behavior_cohort_export validate \
  --publication-root "\$(scripts/py -c '
import sys
from fisheye.analytics_exports.validated_behavior_cohort import read_validated_behavior_export_plan
from fisheye.analytics_exports.validated_behavior_profiles import profile_id_from_record, resolve_validated_behavior_profile
profile = resolve_validated_behavior_profile(profile_id_from_record(sys.argv[1], record_kind="export plan"))
p, _, _ = read_validated_behavior_export_plan(sys.argv[1], table_specs=profile.table_specs)
print(p["publication_root"])
' "\${PLAN}")" \
  --export-run-id "${EXPORT_RUN_ID}"
printf 'complete\n' >"\${STATUS_FILE}"
JOBSCRIPT
chmod 0755 "$SHARD_SCRIPT" "$FINALIZE_SCRIPT"

array_name="vb_${EXPORT_RUN_ID}[1-${MEMBER_COUNT}]%${MAX_ACTIVE}"
array_cmd=(bsub -J "$array_name" -n 1 -M "${MEM_GB}GB" -W "$WALLTIME")
if [[ -n "$QUEUE" ]]; then array_cmd+=(-q "$QUEUE"); fi
array_cmd+=(-o "${RUN_DIR}/shard.%I.out" -e "${RUN_DIR}/shard.%I.err" "$SHARD_SCRIPT")

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

submit_bsub() {
  local output
  if command -v bsub >/dev/null 2>&1; then
    output="$("$@")"
  else
    local remote
    remote="$(print_command "$@")"
    output="$(ssh "$SUBMIT_HOST" "$remote")"
  fi
  printf '%s\n' "$output" >&2
  local job_id
  job_id="$(sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' <<<"$output" | head -n 1)"
  [[ -n "$job_id" ]] || fail "Could not parse submitted LSF job ID"
  printf '%s' "$job_id"
}

if [[ "$SUBMIT" == "0" ]]; then
  printf 'mode=render-only\n'
  printf 'export_run_id=%s\n' "$EXPORT_RUN_ID"
  printf 'member_count=%s\n' "$MEMBER_COUNT"
  printf 'max_active=%s\n' "$MAX_ACTIVE"
  printf 'shard_bsub_command='; print_command "${array_cmd[@]}"
  finalize_cmd=(bsub -J "vb_${EXPORT_RUN_ID}_finalize" -n 1 -M "${MEM_GB}GB" -W "$WALLTIME" -w 'done(<shard-array-job-id>)')
  if [[ -n "$QUEUE" ]]; then finalize_cmd+=(-q "$QUEUE"); fi
  finalize_cmd+=(-o "${RUN_DIR}/finalize.out" -e "${RUN_DIR}/finalize.err" "$FINALIZE_SCRIPT")
  printf 'finalize_bsub_command='; print_command "${finalize_cmd[@]}"
  exit 0
fi

ARRAY_JOB_ID="$(submit_bsub "${array_cmd[@]}")"
finalize_cmd=(bsub -J "vb_${EXPORT_RUN_ID}_finalize" -n 1 -M "${MEM_GB}GB" -W "$WALLTIME" -w "done(${ARRAY_JOB_ID})")
if [[ -n "$QUEUE" ]]; then finalize_cmd+=(-q "$QUEUE"); fi
finalize_cmd+=(-o "${RUN_DIR}/finalize.out" -e "${RUN_DIR}/finalize.err" "$FINALIZE_SCRIPT")
FINALIZE_JOB_ID="$(submit_bsub "${finalize_cmd[@]}")"

{
  printf 'export_run_id=%s\n' "$EXPORT_RUN_ID"
  printf 'plan=%s\n' "$PLAN"
  printf 'palette_repo=%s\n' "$PALETTE_REPO"
  printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  printf 'member_count=%s\n' "$MEMBER_COUNT"
  printf 'max_active=%s\n' "$MAX_ACTIVE"
  printf 'shard_array_job_id=%s\n' "$ARRAY_JOB_ID"
  printf 'finalize_job_id=%s\n' "$FINALIZE_JOB_ID"
} >"$SUBMISSION_FILE"
printf 'mode=submitted\n'
printf 'shard_array_job_id=%s\n' "$ARRAY_JOB_ID"
printf 'finalize_job_id=%s\n' "$FINALIZE_JOB_ID"
printf 'submission_file=%s\n' "$SUBMISSION_FILE"
