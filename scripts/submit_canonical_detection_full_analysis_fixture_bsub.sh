#!/usr/bin/env bash
set -euo pipefail
umask 0002

SPEC=""
DESTINATION=""
BENCHMARK_ROOT=""
RUN_ID=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_ROOT=""
SCRATCH_BASE=""
QUEUE="local"
NCORES=1
MEM_GB=15
WALLTIME="12:00"
PAIR_COPY_MODE="auto"
PREFLIGHT_ONLY=0
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_canonical_detection_full_analysis_fixture_bsub.sh \
  --spec PATH --destination PATH --benchmark-root PATH --run-id ID [options]

Submit one commit-pinned, benchmark-only Crimson full-analysis fixture job.
Plan rendering is the default. The LSF job assembles and validates on node-local
scratch before publishing the paired regular.zarr and hybrid.zarr atomically.

Required:
  --spec PATH              Frozen full-analysis fixture specification
  --destination PATH       Fresh final paired-fixture directory
  --benchmark-root PATH    Approved .palette_benchmarks root
  --run-id ID              Immutable submission/log identifier

Options:
  --palette-repo PATH      Clean cluster-visible commit-pinned Palette checkout
  --submit-host HOST       Citrus SSH poller (default: login1-citrus-poller)
  --log-root PATH          Default: <benchmark-root>/canonical_detection_storage/
                           full_analysis/submissions
  --scratch-base PATH      Existing node-local base override
  --queue NAME             LSF queue (default: local)
  --ncores N               Allocated slots (default: 1)
  --mem-gb N               Approximate total memory request (default: 15)
  --walltime H:MM          Walltime (default: 12:00)
  --pair-copy-mode MODE    auto, copy, or reflink (default: auto)
  --preflight-only         Probe scratch/reflink behavior; do not build a fixture
  --submit                 Submit through LSF; otherwise render only
  -h, --help               Show this help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec) SPEC="$2"; shift 2;;
    --destination) DESTINATION="$2"; shift 2;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --scratch-base) SCRATCH_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --pair-copy-mode) PAIR_COPY_MODE="$2"; shift 2;;
    --preflight-only) PREFLIGHT_ONLY=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$SPEC" ]] || fail "--spec is required"
[[ -n "$DESTINATION" ]] || fail "--destination is required"
[[ -n "$BENCHMARK_ROOT" ]] || fail "--benchmark-root is required"
[[ -n "$RUN_ID" ]] || fail "--run-id is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --run-id"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ "$PAIR_COPY_MODE" =~ ^(auto|copy|reflink)$ ]] || fail "invalid --pair-copy-mode"
[[ -f "$SPEC" ]] || fail "fixture specification not found: $SPEC"
[[ -d "$BENCHMARK_ROOT" ]] || fail "benchmark root not found: $BENCHMARK_ROOT"
[[ ! -e "$DESTINATION" ]] || fail "destination already exists: $DESTINATION"
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain --untracked-files=all)" ]] || \
  fail "Palette checkout must be clean"
if [[ -n "$SCRATCH_BASE" ]]; then
  [[ "$SCRATCH_BASE" != /groups/* ]] || fail "--scratch-base must be node-local"
fi
if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="${BENCHMARK_ROOT%/}/canonical_detection_storage/full_analysis/submissions"
fi
RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
MEM_GB_PER_SLOT=$(( (MEM_GB + NCORES - 1) / NCORES ))
JOB_SCRIPT="${RUN_DIR}/run_fixture_job.sh"
STATUS_FILE="${RUN_DIR}/status.txt"
PREFLIGHT_FILE="${RUN_DIR}/preflight.txt"
RESOURCE_FILE="${RUN_DIR}/resource_usage.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_spec="$(printf '%q' "$SPEC")"
q_destination="$(printf '%q' "$DESTINATION")"
q_benchmark="$(printf '%q' "$BENCHMARK_ROOT")"
q_run_id="$(printf '%q' "$RUN_ID")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_preflight="$(printf '%q' "$PREFLIGHT_FILE")"
q_resource="$(printf '%q' "$RESOURCE_FILE")"
q_scratch="$(printf '%q' "$SCRATCH_BASE")"
q_pair_copy="$(printf '%q' "$PAIR_COPY_MODE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
SPEC=${q_spec}
DESTINATION=${q_destination}
BENCHMARK_ROOT=${q_benchmark}
RUN_ID=${q_run_id}
EXPECTED_COMMIT=${q_expected}
STATUS_FILE=${q_status}
PREFLIGHT_FILE=${q_preflight}
RESOURCE_FILE=${q_resource}
CONFIGURED_SCRATCH_BASE=${q_scratch}
PAIR_COPY_MODE=${q_pair_copy}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY}

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
scratch_root="\${scratch_base}/canonical_detection_full_analysis_\${RUN_ID}"
[[ ! -e "\${scratch_root}" ]] || { printf 'Refusing existing scratch root.\n' >&2; exit 2; }
mkdir -p "\${scratch_root}"
export PYTHONPYCACHEPREFIX="\${scratch_root}/pycache"
interrupted() {
  signal_name="\$1"
  status_tmp="\${STATUS_FILE}.tmp.\$\$"
  {
    printf 'status=interrupted\n'
    printf 'signal=%s\n' "\${signal_name}"
    printf 'interrupted_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "\$(hostname)"
    printf 'job_id=%s\n' "\${LSB_JOBID}"
    printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
    printf 'scratch_root=%s\n' "\${scratch_root}"
  } >"\${status_tmp}"
  mv "\${status_tmp}" "\${STATUS_FILE}"
  exit 143
}
trap 'interrupted TERM' TERM
trap 'interrupted INT' INT
trap 'interrupted HUP' HUP

probe_source="\${scratch_root}/.reflink_probe_source"
probe_clone="\${scratch_root}/.reflink_probe_clone"
dd if=/dev/zero of="\${probe_source}" bs=1M count=64 status=none
set +e
cp --reflink=always -- "\${probe_source}" "\${probe_clone}" 2>"\${scratch_root}/reflink.stderr"
reflink_rc=\$?
set -e
reflink_supported=false
inode_distinct=false
mutation_isolated=false
source_inode=""
clone_inode=""
reflink_stderr="\$(tr '\n' ' ' <"\${scratch_root}/reflink.stderr")"
if [[ "\${reflink_rc}" == "0" ]]; then
  source_inode="\$(stat -c %i "\${probe_source}")"
  clone_inode="\$(stat -c %i "\${probe_clone}")"
  if [[ "\${source_inode}" != "\${clone_inode}" ]]; then inode_distinct=true; fi
  printf 'palette-isolated-mutation' >"\${probe_clone}"
  if [[ "\$(stat -c %s "\${probe_source}")" == "67108864" ]]; then mutation_isolated=true; fi
  if [[ "\${inode_distinct}" == "true" && "\${mutation_isolated}" == "true" ]]; then
    reflink_supported=true
  fi
fi
available_bytes="\$(df -B1 --output=avail "\${scratch_root}" | tail -n 1 | tr -d ' ')"
filesystem_type="\$(stat -f -c %T "\${scratch_root}")"
{
  printf 'status=complete\n'
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'scratch_root=%s\n' "\${scratch_root}"
  printf 'filesystem_type=%s\n' "\${filesystem_type}"
  printf 'available_bytes=%s\n' "\${available_bytes}"
  printf 'reflink_returncode=%s\n' "\${reflink_rc}"
  printf 'reflink_stderr=%s\n' "\${reflink_stderr}"
  printf 'reflink_supported=%s\n' "\${reflink_supported}"
  printf 'source_inode=%s\n' "\${source_inode}"
  printf 'clone_inode=%s\n' "\${clone_inode}"
  printf 'inode_distinct=%s\n' "\${inode_distinct}"
  printf 'mutation_isolated=%s\n' "\${mutation_isolated}"
} >"\${PREFLIGHT_FILE}"
rm -f -- "\${probe_source}" "\${probe_clone}" "\${scratch_root}/reflink.stderr"

if [[ "\${PREFLIGHT_ONLY}" == "1" ]]; then
  rm -rf -- "\${scratch_root}"
  {
    printf 'status=complete\n'
    printf 'mode=preflight_only\n'
    printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "\$(hostname)"
    printf 'job_id=%s\n' "\${LSB_JOBID}"
    printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
    printf 'preflight=%s\n' "\${PREFLIGHT_FILE}"
  } >"\${STATUS_FILE}"
  exit 0
fi

cmd=(
  scripts/py -m fisheye.diagnostics.build_canonical_detection_full_analysis_fixtures
  --spec "\${SPEC}"
  --destination "\${DESTINATION}"
  --benchmark-root "\${BENCHMARK_ROOT}"
  --scratch-root "\${scratch_root}"
  --expected-palette-commit "\${EXPECTED_COMMIT}"
  --pair-copy-mode "\${PAIR_COPY_MODE}"
  --apply
)
printf 'fixture_command='; printf '%q ' "\${cmd[@]}"; printf '\n'
set +e
/usr/bin/time -v -o "\${RESOURCE_FILE}" "\${cmd[@]}"
payload_rc=\$?
set -e

status_tmp="\${STATUS_FILE}.tmp.\$\$"
{
  if [[ "\${payload_rc}" == "0" ]]; then printf 'status=complete\n'; else printf 'status=failed\n'; fi
  printf 'mode=fixture_publication\n'
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'destination=%s\n' "\${DESTINATION}"
  printf 'scratch_root=%s\n' "\${scratch_root}"
  printf 'preflight=%s\n' "\${PREFLIGHT_FILE}"
  printf 'resource_usage=%s\n' "\${RESOURCE_FILE}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
if [[ "\${payload_rc}" == "0" ]]; then rm -rf -- "\${scratch_root}"; fi
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "canonical_detection_full_analysis_${RUN_ID}"
  -q "$QUEUE"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_GB_PER_SLOT}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'preflight_only=%s\n' "$PREFLIGHT_ONLY"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'run_id=%s\n' "$RUN_ID"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'status_file=%s\n' "$STATUS_FILE"
printf 'preflight_file=%s\n' "$PREFLIGHT_FILE"
printf 'resource_file=%s\n' "$RESOURCE_FILE"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and submit host empty"
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
    printf 'preflight_file=%s\n' "$PREFLIGHT_FILE"
    printf 'resource_file=%s\n' "$RESOURCE_FILE"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
