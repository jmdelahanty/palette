#!/usr/bin/env bash
set -euo pipefail

TASK=""
PALETTE_REPO=""
PALETTE_COMMIT=""
RUN_ROOT=""
RUN_ID=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
QUEUE=""
NCORES=4
MEM_GB=16
MAX_ACTIVE=8
WALLTIME="6:00"
COPY_BACKEND="rsync"
ARRAY_INDICES=""
SUBMIT=0

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

usage() {
  cat <<'USAGE'
Usage: submit_composable_chaser_successors_bsub.sh --task PATH \
  --palette-repo PATH --palette-commit SHA --run-root PATH [options]

Render or submit one CPU LSF array task per recording in a frozen
palette.composable_chaser_successor_cohort_task. Each worker revalidates exact
input metadata and runs all recording-local publications serially. Publications
and plot receipts remain selector-ineligible; workers never update SQLite or a
production selector.

Required:
  --task PATH             Frozen cohort task JSON.
  --palette-repo PATH     Clean, cluster-visible, commit-pinned Palette worktree.
  --palette-commit SHA    Full 40-character commit at --palette-repo.
  --run-root PATH         Durable task copy, receipts, and LSF logs.

Options:
  --run-id ID             Stable submission ID (default: UTC timestamp).
  --submit-host HOST      SSH host used when bsub is unavailable locally.
  --queue NAME            LSF queue (default: cluster default).
  --ncores N              CPU slots per recording (default: 4).
  --mem-gb N              Memory request per recording in GB (default: 16).
  --max-active N          Maximum simultaneous array workers (default: 8).
  --walltime H:MM         Per-recording wall time (default: 6:00).
  --copy-backend NAME     rsync or python (default: rsync).
  --array-indices SPEC    Optional LSF indices/ranges, e.g. 1-76,81-84.
                          Default submits every recording index.
  --submit                Submit the array. Default is a no-write rendering.
  -h, --help              Show this help.
USAGE
}

while (($#)); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --palette-repo) PALETTE_REPO="$2"; shift 2 ;;
    --palette-commit) PALETTE_COMMIT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --submit-host) SUBMIT_HOST="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --ncores) NCORES="$2"; shift 2 ;;
    --mem-gb) MEM_GB="$2"; shift 2 ;;
    --max-active) MAX_ACTIVE="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --copy-backend) COPY_BACKEND="$2"; shift 2 ;;
    --array-indices) ARRAY_INDICES="$2"; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$TASK" ]] || fail "--task is required"
[[ -f "$TASK" ]] || fail "Cohort task does not exist: $TASK"
[[ -n "$PALETTE_REPO" ]] || fail "--palette-repo is required"
[[ "$PALETTE_REPO" = /* ]] || fail "--palette-repo must be absolute"
[[ -f "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is absent"
[[ -n "$PALETTE_COMMIT" ]] || fail "--palette-commit is required"
[[ "$PALETTE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "--palette-commit must be a full lowercase SHA"
[[ -n "$RUN_ROOT" ]] || fail "--run-root is required"
[[ "$RUN_ROOT" = /* ]] || fail "--run-root must be absolute"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ "$MAX_ACTIVE" =~ ^[1-9][0-9]*$ ]] || fail "--max-active must be positive"
[[ "$WALLTIME" =~ ^[0-9]+:[0-5][0-9]$ ]] || fail "--walltime must use H:MM"
[[ "$COPY_BACKEND" = "rsync" || "$COPY_BACKEND" = "python" ]] || fail "Unsupported --copy-backend"

OBSERVED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
[[ "$OBSERVED_COMMIT" = "$PALETTE_COMMIT" ]] || fail "Palette HEAD differs from --palette-commit"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || fail "Palette worktree must be clean"

VALIDATION_JSON="$(PYTHONPATH="$PALETTE_REPO/src" "$PALETTE_REPO/scripts/py" -m fisheye.utils.materialize_composable_chaser_successor_cohort validate "$TASK")"
RECORDING_COUNT="$(printf '%s' "$VALIDATION_JSON" | "$PALETTE_REPO/scripts/py" -c 'import json,sys; print(json.load(sys.stdin)["recording_count"])')"
TASK_SHA256="$(printf '%s' "$VALIDATION_JSON" | "$PALETTE_REPO/scripts/py" -c 'import json,sys; print(json.load(sys.stdin)["task_sha256"])')"
[[ "$RECORDING_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "Frozen task has no recordings"

if [[ -z "$ARRAY_INDICES" ]]; then
  ARRAY_INDICES="1-${RECORDING_COUNT}"
fi
[[ "$ARRAY_INDICES" =~ ^[1-9][0-9]*(-[1-9][0-9]*)?(,[1-9][0-9]*(-[1-9][0-9]*)?)*$ ]] \
  || fail "--array-indices must contain positive comma-separated indices/ranges"
SELECTED_RECORDING_COUNT="$("$PALETTE_REPO/scripts/py" -c '
import sys
limit = int(sys.argv[1])
seen = set()
for item in sys.argv[2].split(","):
    bounds = [int(value) for value in item.split("-")]
    start = bounds[0]
    end = bounds[-1]
    if start > end or end > limit:
        raise SystemExit("array index range leaves the frozen task")
    for index in range(start, end + 1):
        if index in seen:
            raise SystemExit("array index ranges overlap")
        seen.add(index)
print(len(seen))
' "$RECORDING_COUNT" "$ARRAY_INDICES")"
[[ "$SELECTED_RECORDING_COUNT" =~ ^[1-9][0-9]*$ ]] \
  || fail "--array-indices selects no recordings"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--run-id contains unsupported characters"

RUN_DIR="${RUN_ROOT%/}/composable_chaser_successors_${RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "Submission run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/receipts"
FROZEN_TASK="$RUN_DIR/cohort_task.json"
cp -- "$TASK" "$FROZEN_TASK"

COPIED_VALIDATION="$(PYTHONPATH="$PALETTE_REPO/src" "$PALETTE_REPO/scripts/py" -m fisheye.utils.materialize_composable_chaser_successor_cohort validate "$FROZEN_TASK")"
COPIED_TASK_SHA256="$(printf '%s' "$COPIED_VALIDATION" | "$PALETTE_REPO/scripts/py" -c 'import json,sys; print(json.load(sys.stdin)["task_sha256"])')"
[[ "$COPIED_TASK_SHA256" = "$TASK_SHA256" ]] || fail "Copied task digest changed"

JOB_SCRIPT="$RUN_DIR/run_one_recording.sh"
cat >"$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
[[ -n "\${LSB_JOBINDEX:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
TASK_INDEX="\${LSB_JOBINDEX}"
SCRATCH_BASE="\${LSB_JOB_TMPDIR:-\${TMPDIR:-/tmp}}/palette_composable_chaser_${RUN_ID}_\${TASK_INDEX}"
export PYTHONPATH=$(printf '%q' "$PALETTE_REPO/src")
export MPLCONFIGDIR="\${SCRATCH_BASE}/matplotlib"
mkdir -p "\${SCRATCH_BASE}" "\${MPLCONFIGDIR}"
$(printf '%q' "$PALETTE_REPO/scripts/py") -m fisheye.utils.materialize_composable_chaser_successor_cohort run-one \
  $(printf '%q' "$FROZEN_TASK") \
  --task-index "\${TASK_INDEX}" \
  --palette-repo $(printf '%q' "$PALETTE_REPO") \
  --palette-commit $(printf '%q' "$PALETTE_COMMIT") \
  --scratch-root "\${SCRATCH_BASE}" \
  --receipt-root $(printf '%q' "$RUN_DIR/receipts") \
  --copy-backend $(printf '%q' "$COPY_BACKEND") \
  --apply
EOF
chmod 0755 "$JOB_SCRIPT"

MEM_MB="$((MEM_GB * 1024))"
BSUB_ARGS=(
  -J "chaser_successors_${RUN_ID}[${ARRAY_INDICES}]%${MAX_ACTIVE}"
  -n "$NCORES"
  -M "$MEM_MB"
  -R "rusage[mem=${MEM_MB}]"
  -W "$WALLTIME"
  -o "$RUN_DIR/logs/%J_%I.out"
  -e "$RUN_DIR/logs/%J_%I.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

{
  printf 'schema_id=palette.composable_chaser_successor_bsub_submission.v1\n'
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'task_sha256=%s\n' "$TASK_SHA256"
  printf 'recording_count=%s\n' "$RECORDING_COUNT"
  printf 'array_indices=%s\n' "$ARRAY_INDICES"
  printf 'selected_recording_count=%s\n' "$SELECTED_RECORDING_COUNT"
  printf 'palette_repo=%s\n' "$PALETTE_REPO"
  printf 'palette_commit=%s\n' "$PALETTE_COMMIT"
  printf 'selector_eligible=false\n'
  printf 'production_authority=false\n'
  printf 'registry_update=false\n'
  printf 'submit_requested=%s\n' "$SUBMIT"
} >"$RUN_DIR/submission.env"

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'task_sha256=%s\n' "$TASK_SHA256"
printf 'recording_count=%s\n' "$RECORDING_COUNT"
printf 'array_indices=%s\n' "$ARRAY_INDICES"
printf 'selected_recording_count=%s\n' "$SELECTED_RECORDING_COUNT"
printf 'palette_commit=%s\n' "$PALETTE_COMMIT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" -eq 0 ]]; then
  printf 'mode=dry_run_no_submission\n'
  exit 0
fi

if command -v bsub >/dev/null 2>&1; then
  SUBMIT_OUTPUT="$("${BSUB_COMMAND[@]}")"
  SUBMIT_MODE="local_bsub"
else
  [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and --submit-host is empty"
  QUOTED_COMMAND="$(printf '%q ' "${BSUB_COMMAND[@]}")"
  SUBMIT_OUTPUT="$(ssh "$SUBMIT_HOST" "$QUOTED_COMMAND")"
  SUBMIT_MODE="ssh_bsub"
fi
printf '%s\n' "$SUBMIT_OUTPUT"
JOB_ID="$(printf '%s\n' "$SUBMIT_OUTPUT" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
[[ -n "$JOB_ID" ]] || fail "Could not parse LSF job ID"
{
  printf 'submit_mode=%s\n' "$SUBMIT_MODE"
  printf 'job_id=%s\n' "$JOB_ID"
} >>"$RUN_DIR/submission.env"
printf 'submit_mode=%s\njob_id=%s\n' "$SUBMIT_MODE" "$JOB_ID"
