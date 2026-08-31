#!/usr/bin/env bash
set -euo pipefail

TASK=""
PALETTE_REPO=""
PALETTE_COMMIT=""
RUN_ROOT=""
RUN_ID=""
PHASE=""
PROOF_ROOT=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
QUEUE=""
NCORES=""
MEM_GB=""
MAX_ACTIVE=8
WALLTIME=""
COPY_BACKEND=rsync
ARRAY_INDICES=""
SUBMIT=0

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

usage() {
  cat <<'USAGE'
Usage: submit_eye_gaze_prerequisites_bsub.sh --phase prove|materialize \
  --task PATH --palette-repo PATH --palette-commit SHA --run-root PATH [options]

Runs one recording per LSF array worker from a frozen
palette.eye_gaze_prerequisite_cohort_task. The prove phase is exhaustive and
read-only. The materialize phase requires those exact commit-bound proofs and
publishes only immutable selector-ineligible repair, subject-shape, and
eye-angle candidates plus numeric validation/review artifacts.

Required:
  --phase NAME            prove or materialize.
  --task PATH             Frozen prerequisite cohort task.
  --palette-repo PATH     Clean cluster-visible commit-pinned Palette worktree.
  --palette-commit SHA    Full commit at --palette-repo.
  --run-root PATH         Durable task, receipt, review, and log root.
  --proof-root PATH       Required for materialize; output proof directory from
                          a completed prove submission.

Options:
  --run-id ID             Stable submission ID (default: UTC timestamp).
  --submit-host HOST      SSH host when bsub is unavailable locally.
  --queue NAME            Optional LSF queue.
  --ncores N              Slots per worker (prove: 1; materialize: 8).
  --mem-gb N              Memory per worker (prove: 8; materialize: 64).
  --max-active N          Simultaneous workers (default: 8).
  --walltime H:MM         Per-worker limit (prove: 2:00; materialize: 8:00).
  --copy-backend NAME     rsync or python (default: rsync; materialize only).
  --array-indices SPEC    Optional indices/ranges, e.g. 1 or 2-84.
  --submit                Submit; omission renders a no-job preview.
  -h, --help              Show this help.
USAGE
}

while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --palette-repo) PALETTE_REPO="$2"; shift 2 ;;
    --palette-commit) PALETTE_COMMIT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --proof-root) PROOF_ROOT="$2"; shift 2 ;;
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

[[ "$PHASE" = prove || "$PHASE" = materialize ]] || fail "--phase must be prove or materialize"
[[ -f "$TASK" ]] || fail "--task must name an existing file"
[[ "$PALETTE_REPO" = /* && -f "$PALETTE_REPO/scripts/py" ]] || fail "--palette-repo must be an absolute Palette checkout"
[[ "$PALETTE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "--palette-commit must be a full lowercase SHA"
[[ "$RUN_ROOT" = /* ]] || fail "--run-root must be absolute"
[[ "$COPY_BACKEND" = rsync || "$COPY_BACKEND" = python ]] || fail "unsupported --copy-backend"
if [[ "$PHASE" = materialize ]]; then
  [[ "$PROOF_ROOT" = /* && -d "$PROOF_ROOT" ]] || fail "materialize requires an absolute existing --proof-root"
fi

if [[ "$PHASE" = prove ]]; then
  NCORES="${NCORES:-1}"
  MEM_GB="${MEM_GB:-8}"
  WALLTIME="${WALLTIME:-2:00}"
else
  NCORES="${NCORES:-8}"
  MEM_GB="${MEM_GB:-64}"
  WALLTIME="${WALLTIME:-8:00}"
fi
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ "$MAX_ACTIVE" =~ ^[1-9][0-9]*$ ]] || fail "--max-active must be positive"
[[ "$WALLTIME" =~ ^[0-9]+:[0-5][0-9]$ ]] || fail "--walltime must use H:MM"

OBSERVED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
[[ "$OBSERVED_COMMIT" = "$PALETTE_COMMIT" ]] || fail "Palette HEAD differs from --palette-commit"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || fail "Palette worktree must be clean"

VALIDATION_JSON="$(PYTHONPATH="$PALETTE_REPO/src" "$PALETTE_REPO/scripts/py" -m fisheye.utils.materialize_eye_gaze_prerequisite_cohort validate "$TASK")"
RECORDING_COUNT="$(printf '%s' "$VALIDATION_JSON" | "$PALETTE_REPO/scripts/py" -c 'import json,sys; print(json.load(sys.stdin)["recording_count"])')"
TASK_SHA256="$(printf '%s' "$VALIDATION_JSON" | "$PALETTE_REPO/scripts/py" -c 'import json,sys; print(json.load(sys.stdin)["task_sha256"])')"
[[ "$RECORDING_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "frozen task has no recordings"

if [[ -z "$ARRAY_INDICES" ]]; then
  ARRAY_INDICES="1-${RECORDING_COUNT}"
fi
[[ "$ARRAY_INDICES" =~ ^[1-9][0-9]*(-[1-9][0-9]*)?(,[1-9][0-9]*(-[1-9][0-9]*)?)*$ ]] || fail "invalid --array-indices"
SELECTED_COUNT="$("$PALETTE_REPO/scripts/py" -c '
import sys
limit = int(sys.argv[1])
seen = set()
for item in sys.argv[2].split(","):
    bounds = [int(value) for value in item.split("-")]
    start, stop = bounds[0], bounds[-1]
    if start > stop or stop > limit:
        raise SystemExit("array range leaves task")
    for index in range(start, stop + 1):
        if index in seen:
            raise SystemExit("array ranges overlap")
        seen.add(index)
print(len(seen))
' "$RECORDING_COUNT" "$ARRAY_INDICES")"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--run-id contains unsupported characters"
RUN_DIR="${RUN_ROOT%/}/eye_gaze_prerequisites_${PHASE}_${RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory exists: $RUN_DIR"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/receipts" "$RUN_DIR/reviews"
FROZEN_TASK="$RUN_DIR/cohort_task.json"
cp -- "$TASK" "$FROZEN_TASK"
COPIED_VALIDATION="$(PYTHONPATH="$PALETTE_REPO/src" "$PALETTE_REPO/scripts/py" -m fisheye.utils.materialize_eye_gaze_prerequisite_cohort validate "$FROZEN_TASK")"
COPIED_SHA="$(printf '%s' "$COPIED_VALIDATION" | "$PALETTE_REPO/scripts/py" -c 'import json,sys; print(json.load(sys.stdin)["task_sha256"])')"
[[ "$COPIED_SHA" = "$TASK_SHA256" ]] || fail "copied task digest changed"

if [[ "$PHASE" = prove ]]; then
  PROOF_ROOT="$RUN_DIR/proofs"
  mkdir -p "$PROOF_ROOT"
fi

JOB_SCRIPT="$RUN_DIR/run_one_recording.sh"
cat >"$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
[[ -n "\${LSB_JOBINDEX:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
TASK_INDEX="\${LSB_JOBINDEX}"
SCRATCH_BASE="\${LSB_JOB_TMPDIR:-\${TMPDIR:-/tmp}}/palette_eye_gaze_${PHASE}_${RUN_ID}_\${TASK_INDEX}"
export PYTHONPATH=$(printf '%q' "$PALETTE_REPO/src")
export MPLCONFIGDIR="\${SCRATCH_BASE}/matplotlib"
mkdir -p "\${SCRATCH_BASE}" "\${MPLCONFIGDIR}"
EOF
if [[ "$PHASE" = prove ]]; then
  cat >>"$JOB_SCRIPT" <<EOF
$(printf '%q' "$PALETTE_REPO/scripts/py") -m fisheye.utils.materialize_eye_gaze_prerequisite_cohort prove-one \
  $(printf '%q' "$FROZEN_TASK") \
  --task-index "\${TASK_INDEX}" \
  --palette-repo $(printf '%q' "$PALETTE_REPO") \
  --palette-commit $(printf '%q' "$PALETTE_COMMIT") \
  --proof-root $(printf '%q' "$PROOF_ROOT")
EOF
else
  cat >>"$JOB_SCRIPT" <<EOF
$(printf '%q' "$PALETTE_REPO/scripts/py") -m fisheye.utils.materialize_eye_gaze_prerequisite_cohort run-one \
  $(printf '%q' "$FROZEN_TASK") \
  --task-index "\${TASK_INDEX}" \
  --palette-repo $(printf '%q' "$PALETTE_REPO") \
  --palette-commit $(printf '%q' "$PALETTE_COMMIT") \
  --proof-root $(printf '%q' "$PROOF_ROOT") \
  --scratch-root "\${SCRATCH_BASE}" \
  --receipt-root $(printf '%q' "$RUN_DIR/receipts") \
  --copy-backend $(printf '%q' "$COPY_BACKEND") \
  --num-workers $(printf '%q' "$NCORES") \
  --apply
EOF
fi
chmod 0755 "$JOB_SCRIPT"

MEM_MB="$((MEM_GB * 1024))"
BSUB_ARGS=(
  -J "eye_gaze_${PHASE}_${RUN_ID}[${ARRAY_INDICES}]%${MAX_ACTIVE}"
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
  printf 'schema_id=palette.eye_gaze_prerequisite_bsub_submission.v1\n'
  printf 'phase=%s\n' "$PHASE"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'task_sha256=%s\n' "$TASK_SHA256"
  printf 'recording_count=%s\n' "$RECORDING_COUNT"
  printf 'array_indices=%s\n' "$ARRAY_INDICES"
  printf 'selected_recording_count=%s\n' "$SELECTED_COUNT"
  printf 'palette_repo=%s\n' "$PALETTE_REPO"
  printf 'palette_commit=%s\n' "$PALETTE_COMMIT"
  printf 'proof_root=%s\n' "$PROOF_ROOT"
  printf 'selector_eligible=false\nproduction_authority=false\nregistry_update=false\nselector_activation=false\n'
  printf 'submit_requested=%s\n' "$SUBMIT"
} >"$RUN_DIR/submission.env"

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'task_sha256=%s\n' "$TASK_SHA256"
printf 'recording_count=%s\n' "$RECORDING_COUNT"
printf 'array_indices=%s\n' "$ARRAY_INDICES"
printf 'selected_recording_count=%s\n' "$SELECTED_COUNT"
printf 'proof_root=%s\n' "$PROOF_ROOT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" -eq 0 ]]; then
  printf 'mode=dry_run_no_submission\n'
  exit 0
fi
if command -v bsub >/dev/null 2>&1; then
  SUBMIT_OUTPUT="$("${BSUB_COMMAND[@]}")"
  SUBMIT_MODE=local_bsub
else
  [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and --submit-host is empty"
  QUOTED_COMMAND="$(printf '%q ' "${BSUB_COMMAND[@]}")"
  SUBMIT_OUTPUT="$(ssh "$SUBMIT_HOST" "$QUOTED_COMMAND")"
  SUBMIT_MODE=ssh_bsub
fi
printf '%s\n' "$SUBMIT_OUTPUT"
JOB_ID="$(printf '%s\n' "$SUBMIT_OUTPUT" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
[[ -n "$JOB_ID" ]] || fail "could not parse LSF job ID"
printf 'submit_mode=%s\njob_id=%s\n' "$SUBMIT_MODE" "$JOB_ID" | tee -a "$RUN_DIR/submission.env"
