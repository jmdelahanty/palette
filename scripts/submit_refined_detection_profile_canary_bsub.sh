#!/usr/bin/env bash
set -euo pipefail
umask 0002

SOURCE_GROUP=""
SOURCE_FIXTURE_MANIFEST=""
SOURCE_RUN_ID=""
RECORDING_IDENTITY=""
DESTINATION=""
CANARY_ID=""
CRIMSON_IMPLEMENTATION_COMMIT=""
CRIMSON_EVIDENCE_COMMIT=""
CRIMSON_EVIDENCE_SHA256=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_ROOT=""
SCRATCH_BASE=""
QUEUE="local"
MEM_GB=8
WALLTIME="02:00"
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_refined_detection_profile_canary_bsub.sh [required options]

Render or submit one commit-pinned full-duration refined-detection profile
canary. The job copies its immutable source to node-local scratch, constructs
the regular and access-aware stores there, then verifies and atomically copies
each candidate back to a fresh .palette_benchmarks workflow.

Required:
  --source-group PATH
  --source-fixture-manifest PATH
  --source-run-id ID
  --recording-identity ID
  --destination PATH
  --canary-id ID
  --crimson-implementation-commit SHA
  --crimson-evidence-commit SHA
  --crimson-evidence-sha256 SHA256

Options:
  --palette-repo PATH   Clean commit-pinned cluster-visible checkout
  --submit-host HOST    Citrus poller (default: login1-citrus-poller)
  --log-root PATH       Default: <destination-parent>/submissions
  --scratch-base PATH   Existing node-local scratch base
  --queue NAME          LSF queue (default: local)
  --mem-gb N            Total requested memory (default: 8)
  --walltime H:MM       Walltime (default: 02:00)
  --submit              Submit; otherwise render only
  -h, --help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-group) SOURCE_GROUP="$2"; shift 2;;
    --source-fixture-manifest) SOURCE_FIXTURE_MANIFEST="$2"; shift 2;;
    --source-run-id) SOURCE_RUN_ID="$2"; shift 2;;
    --recording-identity) RECORDING_IDENTITY="$2"; shift 2;;
    --destination) DESTINATION="$2"; shift 2;;
    --canary-id) CANARY_ID="$2"; shift 2;;
    --crimson-implementation-commit) CRIMSON_IMPLEMENTATION_COMMIT="$2"; shift 2;;
    --crimson-evidence-commit) CRIMSON_EVIDENCE_COMMIT="$2"; shift 2;;
    --crimson-evidence-sha256) CRIMSON_EVIDENCE_SHA256="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --scratch-base) SCRATCH_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -d "$SOURCE_GROUP" ]] || fail "source group not found: $SOURCE_GROUP"
[[ -f "$SOURCE_FIXTURE_MANIFEST" ]] || \
  fail "source fixture manifest not found: $SOURCE_FIXTURE_MANIFEST"
[[ -n "$SOURCE_RUN_ID" ]] || fail "--source-run-id is required"
[[ -n "$RECORDING_IDENTITY" ]] || fail "--recording-identity is required"
[[ -n "$DESTINATION" ]] || fail "--destination is required"
[[ -n "$CANARY_ID" ]] || fail "--canary-id is required"
[[ "$CANARY_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe canary ID"
[[ "$CRIMSON_IMPLEMENTATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  fail "invalid Crimson implementation commit"
[[ "$CRIMSON_EVIDENCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  fail "invalid Crimson evidence commit"
[[ "$CRIMSON_EVIDENCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  fail "invalid Crimson evidence SHA-256"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ "$DESTINATION" == */.palette_benchmarks/* ]] || \
  fail "destination must be below .palette_benchmarks"
[[ ! -e "$DESTINATION" ]] || fail "destination already exists: $DESTINATION"
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain --untracked-files=all)" ]] || \
  fail "Palette checkout must be clean"
if [[ -n "$SCRATCH_BASE" ]]; then
  [[ -d "$SCRATCH_BASE" ]] || fail "scratch base not found: $SCRATCH_BASE"
  [[ "$SCRATCH_BASE" != /groups/* ]] || fail "scratch base must be node-local"
fi

if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$(dirname -- "$DESTINATION")/submissions"
fi
RUN_DIR="${LOG_ROOT%/}/${CANARY_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="$RUN_DIR/run_canary_job.sh"
STATUS_FILE="$RUN_DIR/status.txt"
RESOURCE_FILE="$RUN_DIR/resource_usage.txt"
SUBMISSION_FILE="$RUN_DIR/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_source="$(printf '%q' "$SOURCE_GROUP")"
q_fixture="$(printf '%q' "$SOURCE_FIXTURE_MANIFEST")"
q_source_run="$(printf '%q' "$SOURCE_RUN_ID")"
q_recording="$(printf '%q' "$RECORDING_IDENTITY")"
q_destination="$(printf '%q' "$DESTINATION")"
q_canary="$(printf '%q' "$CANARY_ID")"
q_crimson_impl="$(printf '%q' "$CRIMSON_IMPLEMENTATION_COMMIT")"
q_crimson_evidence="$(printf '%q' "$CRIMSON_EVIDENCE_COMMIT")"
q_crimson_sha="$(printf '%q' "$CRIMSON_EVIDENCE_SHA256")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_resource="$(printf '%q' "$RESOURCE_FILE")"
q_scratch="$(printf '%q' "$SCRATCH_BASE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
SOURCE_GROUP=${q_source}
SOURCE_FIXTURE_MANIFEST=${q_fixture}
SOURCE_RUN_ID=${q_source_run}
RECORDING_IDENTITY=${q_recording}
DESTINATION=${q_destination}
CANARY_ID=${q_canary}
CRIMSON_IMPLEMENTATION_COMMIT=${q_crimson_impl}
CRIMSON_EVIDENCE_COMMIT=${q_crimson_evidence}
CRIMSON_EVIDENCE_SHA256=${q_crimson_sha}
EXPECTED_COMMIT=${q_expected}
STATUS_FILE=${q_status}
RESOURCE_FILE=${q_resource}
CONFIGURED_SCRATCH_BASE=${q_scratch}

[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || {
  printf 'Palette commit mismatch.\n' >&2
  exit 2
}
[[ -z "\$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'Refusing dirty Palette checkout.\n' >&2
  exit 2
}

if [[ -n "\${CONFIGURED_SCRATCH_BASE}" ]]; then
  scratch_base="\${CONFIGURED_SCRATCH_BASE}"
elif [[ -d "/scratch/\${USER}" && -w "/scratch/\${USER}" ]]; then
  scratch_base="/scratch/\${USER}/\${LSB_JOBID}"
elif [[ -n "\${TMPDIR:-}" && -d "\${TMPDIR}" && -w "\${TMPDIR}" ]]; then
  scratch_base="\${TMPDIR}/palette/\${LSB_JOBID}"
else
  printf 'No writable node-local scratch root is available.\n' >&2
  exit 2
fi
case "\${scratch_base}" in /groups/*) printf 'Refusing shared scratch.\n' >&2; exit 2;; esac
scratch_root="\${scratch_base}/refined_detection_profile_\${CANARY_ID}"
[[ ! -e "\${scratch_root}" ]] || { printf 'Refusing existing scratch root.\n' >&2; exit 2; }
mkdir -p "\${scratch_root}"
export PYTHONPYCACHEPREFIX="\${scratch_root}/pycache"

cmd=(
  scripts/py -m fisheye.diagnostics.publish_refined_detection_profile_canary
  --source-group "\${SOURCE_GROUP}"
  --source-fixture-manifest "\${SOURCE_FIXTURE_MANIFEST}"
  --source-run-id "\${SOURCE_RUN_ID}"
  --recording-identity "\${RECORDING_IDENTITY}"
  --destination "\${DESTINATION}"
  --scratch-root "\${scratch_root}"
  --canary-id "\${CANARY_ID}"
  --crimson-implementation-commit "\${CRIMSON_IMPLEMENTATION_COMMIT}"
  --crimson-evidence-commit "\${CRIMSON_EVIDENCE_COMMIT}"
  --crimson-evidence-sha256 "\${CRIMSON_EVIDENCE_SHA256}"
)
set +e
/usr/bin/time -v -o "\${RESOURCE_FILE}" "\${cmd[@]}"
payload_rc=\$?
set -e

status_tmp="\${STATUS_FILE}.tmp.\$\$"
{
  if [[ "\${payload_rc}" == "0" ]]; then printf 'status=complete\n'; else printf 'status=failed\n'; fi
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'destination=%s\n' "\${DESTINATION}"
  printf 'scratch_root=%s\n' "\${scratch_root}"
  printf 'resource_usage=%s\n' "\${RESOURCE_FILE}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
if [[ "\${payload_rc}" == "0" ]]; then rm -rf -- "\${scratch_root}"; fi
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "refined_detection_profile_${CANARY_ID}"
  -q "$QUEUE"
  -n 1
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'status_file=%s\n' "$STATUS_FILE"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_mode="ssh_bsub"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse an LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'job_script=%s\n' "$JOB_SCRIPT"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'resource_file=%s\n' "$RESOURCE_FILE"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
